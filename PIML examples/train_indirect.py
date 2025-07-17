import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import RNN, TimeDistributed, Dense, Rescaling, Conv1D
from numpy.lib.stride_tricks import sliding_window_view
"""
ML model with physics-integrated components does indirect measurement of k
to solve inverse problem.
"""

class SpringMass(Layer):
    
    '''
    Implemented solvers are:
        soln (exact solution assuming constant F)
        rk4 (Runge-Kutta method)
        euler (forward Euler)
    '''
    def __init__(self, dt, m, c, solver='rk4', **kwargs):
        self.dt = dt
        self.m = m
        self.c = c
        self.solver = solver
        self.state_size = [tf.TensorShape([1,]), tf.TensorShape([1,])]
        self.output_size = tf.TensorShape([1,])
        if(solver == 'soln'):
            # precompute rate of decay and its square
            self.r = self.c/(2*self.m)
            self.r2 = self.r**2
        super(SpringMass, self).__init__(**kwargs)
        self.build(input_shape=[2,])
        self.built = True
    
    def get_config(self):
        return {"dt": self.dt, "m": self.m, "c": self.c}
    
    '''
    redirect call to the implementation of the chosen solver
    
    inputs: [y, F] where y is the input to k_call and F is forcing
    
    returns output, [states] as reconstructed xddot [updated xdot, x]
    '''
    def call(self, inputs, states):
        if(self.solver == 'soln'):
            return self.soln(inputs, states)
        elif(self.solver == 'rk4'):
            return self.rk4(inputs, states)
        elif(self.solver == 'euler'):
            return self.euler(inputs, states)
        elif(self.solver == 'lugre'):
            return self.lugre(inputs, states)
    
    '''
    Exact solution to damped harmonic motion assuming F is constant over dt.
    '''
    def soln(self, inputs, states):
        k, F = tf.split(inputs, 2, -1)
        xdot0, x0 = states
        # print('xdot0', xdot0.shape)
        # print('x0', x0.shape)
        dt = self.dt
        omega_d = tf.sqrt(k/self.m - self.r2)
        B = x0
        A = (xdot0 + x0*self.r)/omega_d
        
        # We use autodifferentiation to save calculating xdot
        with tf.GradientTape() as tape:
            tape.watch(dt)
            x = A*tf.sin(omega_d*self.dt) + B*tf.cos(omega_d*self.dt)
            x = tf.exp(-self.r*dt)*x
        xdot = tape.jacobian(x, dt)
        xddot = (-self.c*xdot - k*x + F)/self.m
        # print('xdot', xdot.shape)
        # print('x', x.shape)
        return xddot, [xdot, x]
    
    '''
    Runge-Kutta method.
    '''
    def rk4(self, inputs, states):
        k, F = tf.split(inputs, 2, -1)
        # print('k', k.shape)
        # print('F', F.shape)
        xdot0, x0 = states
        # print('xdot0', xdot0.shape)
        # print('x0', x0.shape)
        dt = self.dt
        halfdt = 0.5*dt
        def f(xdot, x):
            xddot = (-self.c*xdot - k*x + F)/self.m
            
            return xddot, xdot
        
        xdot1, x1 = f(xdot0, x0)
        xdot2, x2 = f(xdot0 + halfdt*xdot1, x0 + halfdt*x1)
        xdot3, x3 = f(xdot0 + halfdt*xdot2, x0 + halfdt*x2)
        xdot4, x4 = f(xdot0 + dt*xdot3, x0 + dt*x3)
        
        xdot = xdot0 + dt/6*(xdot1 + 2*xdot2 + 2*xdot3 + xdot4)
        x = x0 + dt/6*(x1 + 2*x2 + 2*x3 + x4)
        # print('xdot', xdot.shape)
        # print('x', x.shape)
        xddot = (-self.c*xdot - k*x + F)/self.m
        # print('xddot', xddot.shape)
        return xddot, [xdot, x]
    
    '''
    Forward Euler method.
    '''
    def euler(self, inputs, states):
        k, F = tf.split(inputs, 2, -1)
        xdot0, x0 = states
        xddot = 1/self.m*(F - self.c*xdot0 - k*x0)
        
        xdot = self.dt*xddot
        x = self.dt*xdot0
        xddot = -self.c*xdot - k*x + F
        return xddot, [xdot, x]

'''
RNN Layer for using SpringMass layer 
'''
class SpringMassRNN(RNN):
    
    def __init__(self, dt, m, c, solver='rk4',
                 return_sequences=False, stateful=False, **kwargs):
        self.dt = dt
        self.m = m
        self.c = c
        self.solver = solver
        self.cell = SpringMass(dt, m, c, solver=solver)
        # self.output_shape = 
        super(SpringMassRNN, self).__init__(
            self.cell,
            return_sequences=return_sequences,
            stateful=stateful,
            **kwargs
        )
    
    def get_config(self):
        return {"dt": self.dt, "m": self.m, "c": self.c}

"""
The training generator creates a sort-of virtual array so that passing
over the data per epoch his done optimally.
"""
class DatasetGenerator(keras.utils.Sequence):
    
    def __init__(self, x, v, a, F, batch_size=32,
                 train_len=50, y_len=50, conv_N=10, shuffle=True):
        self.x = x
        self.v = v
        self.a = a
        self.F = F
        self.batch_size=batch_size
        self.train_len= train_len
        self.y_len = y_len
        self.conv_N = conv_N
        self.shuffle = shuffle
        
        self.N = x.shape[0]
        self.T = self.x.shape[1] - (self.train_len+self.y_len+self.conv_N) + 2
        self.n_samples = self.N*self.T
        
        self.indices = np.arange(self.n_samples)
        if(self.shuffle):
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.n_samples/self.batch_size))
    
    def __getitem__(self, index):
        inds = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        y_input = np.zeros((self.batch_size, self.train_len+self.conv_N-1, self.y_len))
        F_input = np.zeros((self.batch_size, self.train_len, 1))
        v_init = np.zeros((self.batch_size, 1))
        x_init = np.zeros((self.batch_size, 1))
        a_output = np.zeros((self.batch_size, self.train_len, 1))
        for k, ind in enumerate(inds):
            i = ind // self.T
            j = (ind % self.T) + self.y_len + self.train_len + self.conv_N - 2
            
            v_init[k] = self.v[i, j-self.train_len-1:j-self.train_len]
            x_init[k] = self.x[i,j-self.train_len-1:j-self.train_len]
            F_input[k] = np.expand_dims(self.F[i, j-self.train_len:j], -1)
            a_output[k] = np.expand_dims(self.a[i,j-self.train_len:j], -1)
            y_input[k] = sliding_window_view(self.a[i:i+1,j-self.y_len-self.train_len-self.conv_N+2:j], [self.y_len], axis=1)
        
        return [y_input, F_input, v_init, x_init], a_output
    
    def on_epoch_end(self):
        if(self.shuffle):
            np.random.shuffle(self.indices)
#%%
def main():
    #%%
    # load data
    all_data = np.load('./data/with_friction.npz')['data']
    # downsample by a factor of 20 so that sampling rate it 50 S/s
    all_data = all_data[:,:-1:20,:]
    
    t = all_data[0,:,0]
    x = all_data[:,:,1]
    v = all_data[:,:,2]
    a = all_data[:,:,3]
    k = all_data[:,:,4]
    F = all_data[:,:,5]
    
    # divide train and test
    x_train = x[:80]; x_test = x[80:]
    v_train = v[:80]; v_test = v[80:]
    a_train = a[:80]; a_test = a[80:]
    k_train = k[:80]; k_test = k[80:] # k_train is never used
    F_train = F[:80]; F_test = F[80:]
    
    # normalize x and k
    k_m = np.mean(k); k_std = np.std(k)
    
    # training parameters
    train_len = 50
    batch_size = 32
    conv_N = 10
    epochs = 20
    
    training_generator = DatasetGenerator(x_train, v_train, a_train, F_train,
                                          batch_size=batch_size,
                                          train_len=train_len,
                                          y_len=train_len,
                                          conv_N=conv_N)
    testing_generator = DatasetGenerator(x_test, v_test, a_test, F_test,
                                         batch_size=batch_size,
                                         train_len=train_len,
                                         y_len=train_len,
                                         conv_N=conv_N)
    # system constants
    dt = tf.constant(t[1] - t[0], dtype=tf.float32)
    m = tf.constant(1.0, dtype=tf.float32)
    c = tf.constant(0.2, dtype=tf.float32)
    # construct keras model
    k_model = keras.models.Sequential([
        TimeDistributed(Dense(100, activation='sigmoid', input_shape=[train_len,])),
        TimeDistributed(Dense(100, activation='sigmoid')),
        TimeDistributed(Dense(100, activation='sigmoid')),
        TimeDistributed(Dense(1, activation=None)),
        Conv1D(1, conv_N, strides=1, padding='valid', use_bias=False, trainable=False),
        TimeDistributed(Rescaling(k_std, offset=k_m)),
    ])
    y_input = keras.Input([None, train_len])
    F_input = keras.Input([None, 1])
    xdot_init = keras.Input([1,])
    x_init = keras.Input([1,])
    
    k = k_model(y_input)
    concat = keras.layers.Concatenate()([k, F_input])
    spring_mass = SpringMassRNN(dt, 
                                m=m,
                                c=c,
                                solver='rk4',
                                return_sequences=True,
                                stateful=False,
                                # input_shape=([None, 50],[None, 1])
    )(concat, initial_state=[xdot_init, x_init])
    
    training_model = keras.Model(
        inputs = [y_input, F_input, xdot_init, x_init],
        outputs = spring_mass
    )
    # train model
    adam = keras.optimizers.Adam(
        learning_rate = 0.001
    ) #clipnorm=1
    training_model.compile(
        loss="mse",
        optimizer=adam,
        run_eagerly=False,
    )
    # initialize 1d convolutional layer with mean weights
    k_model.layers[-2].set_weights([np.full((10, 1, 1), 1/10)])
    training_model.fit(
        training_generator,
        epochs=epochs,
        validation_data = testing_generator,
    )
    # evaluating model
    
    k_pred_tot = np.zeros((20, k_test.shape[1]-train_len-conv_N+2))
    # k_pred_tot = np.zeros((20, (k_test.shape[1]-50)//10))
    for i in range(20):
        a_in = np.expand_dims(sliding_window_view(a_test[i], [train_len]), 0)
        k_pred = k_model.predict(a_in)
        k_pred_tot[i] = k_pred.flatten()
    
    
    np.save('./model_predictions/indirect/k_pred_filtered.npy', k_pred_tot)
    k_true = k_test[:,train_len-1:]
    
    # one prediction from the validation set
    # import matplotlib.pyplot as plt
    # i = 0
    # plt.figure(figsize=(5, 4))
    # plt.plot(t[train_len-1:], k_pred_tot[i], c='tab:orange', label='pred stiffness')
    # plt.plot(t[train_len-1:], k_true[i], c='tab:blue', label='true stiffness')
    # plt.xlabel('time (s)')
    # plt.ylabel('stiffness (N/m)')
    # plt.xlim((0, 120))
    # plt.legend()
    # plt.tight_layout()
    #%%

if __name__ == '__main__':
    main()