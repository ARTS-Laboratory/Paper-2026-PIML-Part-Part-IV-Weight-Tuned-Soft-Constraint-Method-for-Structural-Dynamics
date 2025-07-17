import numpy as np
import matplotlib.pyplot as plt

"""
examining data


"""
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
k_train = k[:80]; k_test = k[80:]
F_train = F[:80]; F_test = F[80:]

k_pred = np.load('./model_predictions/indirect/k_pred.npy')

i=3
plt.figure()
plt.plot(t[49:], k_test[i,49:], label='true')
plt.plot(t[49:], k_pred[i], label='predicted')
plt.legend()
plt.tight_layout()

#%% make a new indirect model and feed the predictions and true k to get error
import tensorflow as tf
from tensorflow import keras
from indirect_v4 import SpringMass, SpringMassRNN, DatasetGenerator
from numpy.lib.stride_tricks import sliding_window_view

class DatasetGenerator(keras.utils.Sequence):
    
    def __init__(self, x, v, a, F, k, batch_size=32, train_len=50, y_len=50, shuffle=True):
        self.x = x
        self.v = v
        self.a = a
        self.F = F
        self.k = k
        self.batch_size=batch_size
        self.train_len= train_len
        self.y_len = y_len
        self.shuffle = shuffle
        
        self.N = x.shape[0]
        self.T = self.x.shape[1] - (self.train_len+y_len) + 1
        self.n_samples = self.N*self.T
        
        self.indices = np.arange(self.n_samples)
        if(self.shuffle):
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.n_samples/self.batch_size))
    
    def __getitem__(self, index):
        inds = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        y_input = np.zeros((self.batch_size, self.train_len, self.y_len))
        F_input = np.zeros((self.batch_size, self.train_len, 1))
        k_input = np.zeros((self.batch_size, self.train_len, 1))
        v_init = np.zeros((self.batch_size, 1))
        x_init = np.zeros((self.batch_size, 1))
        a_output = np.zeros((self.batch_size, self.train_len, 1))
        for k, ind in enumerate(inds):
            i = ind // self.T
            j = (ind % self.T) + self.y_len + self.train_len
            
            v_init[k] = self.v[i, j-self.train_len-1:j-self.train_len]
            x_init[k] = self.x[i,j-self.train_len-1:j-self.train_len]
            F_input[k] = np.expand_dims(self.F[i, j-self.train_len:j], -1)
            k_input[k] = np.expand_dims(self.k[i,j-self.train_len:j], -1)
            a_output[k] = np.expand_dims(self.a[i,j-self.train_len:j], -1)
            y_input[k] = sliding_window_view(self.a[i:i+1,j-self.y_len-self.train_len+1:j], [self.y_len], axis=1)
        
        return [y_input, F_input, k_input, v_init, x_init], a_output
    
    def on_epoch_end(self):
        if(self.shuffle):
            np.random.shuffle(self.indices)
            
#%% create the dataset generator
train_len = 50
batch_size = 32
epochs = 20

k_pred = np.append(np.zeros((k_pred.shape[0], 49)), k_pred, axis=1)

# only use one test
i=1
k_pred = k_pred[i:i+1]
k_test = k_test[i:i+1]
x_test = x_test[i:i+1]
v_test = v_test[i:i+1]
a_test = a_test[i:i+1]
F_test = F_test[i:i+1]

testing_generator = DatasetGenerator(x_test, v_test, a_test, F_test, k_pred, batch_size=batch_size, train_len=train_len, y_len=train_len, shuffle=False)

#%%
dt = tf.constant(t[1] - t[0], dtype=tf.float32)
m = tf.constant(1.0, dtype=tf.float32)
c = tf.constant(0.2, dtype=tf.float32)

k_input = keras.Input(shape=[None, 1])
y_input = keras.Input([None, 50])
F_input = keras.Input([None, 1])
xdot_init = keras.Input([1,])
x_init = keras.Input([1,])

k = k_input

concat = keras.layers.Concatenate()([k, F_input])
spring_mass = SpringMassRNN(dt, 
                            m=m,
                            c=c,
                            solver='rk4',
                            return_sequences=True,
                            stateful=False,
                            # input_shape=([None, 50],[None, 1])
)(concat, initial_state=[xdot_init, x_init])

model = keras.Model(
    inputs = [y_input, F_input, k_input, xdot_init, x_init],
    outputs = spring_mass
)

a_output = model.predict(testing_generator)

#%%
i=1300
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t[:train_len], a_test[0,i:i+train_len], label='true')
ax1.plot(t[:train_len], a_output[i].flatten(), label='predicted')
ax1.legend()
ax2.plot(t[:train_len], k_test[0,i+train_len:i+2*train_len], label='true')
ax2.plot(t[:train_len], k_pred[0,i+train_len:i+2*train_len], label='predicted')
ax2.legend()
plt.tight_layout()

mse = np.mean(np.square(a_test[0,i:i+train_len] - a_output[i].flatten()))
