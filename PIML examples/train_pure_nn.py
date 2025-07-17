import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from numpy.lib.stride_tricks import sliding_window_view
"""
Pure data-driven approach with neural networks.
"""
#%%
"""
The training generator creates a sort-of virtual array so that passing
over the data per epoch his done optimally.

returns batches
"""
class DatasetGenerator(keras.utils.Sequence):
    
    def __init__(self, x, y, batch_size=32, train_len=50, shuffle=True):
        self.x = x
        self.y = y
        self.train_len= train_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.N = self.x.shape[0]
        self.T = self.x.shape[1] - self.train_len
        self.n_samples = self.N*self.T
        
        self.indices = np.arange(self.n_samples)
        if(self.shuffle):
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.n_samples/self.batch_size))
    
    def __getitem__(self, index):
        inds = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X = np.zeros((self.batch_size, self.train_len))
        Y = np.zeros(self.batch_size)
        for k, ind in enumerate(inds):
            i = ind // self.T
            j = ind % self.T
            X[k] = self.x[i,j:j+self.train_len]
            Y[k] = self.y[i,j+self.train_len]
        
        return X, Y
    
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
    a = all_data[:,:,3]
    k = all_data[:,:,4]
    
    # the model only uses a and k
    a_train = a[:80]; a_test = a[80:]
    k_train = k[:80]; k_test = k[80:]
    
    # normalize x and k
    a_m = np.mean(a); a_std = np.std(a)
    k_m = np.mean(k); k_std = np.std(k)
    X_train = (a_train - a_m)/a_std
    X_test = (a_test - a_m)/a_std
    Y_train = (k_train - k_m)/k_std
    Y_test = (k_test - k_m)/k_std
    
    # training parameters
    train_len = 50
    batch_size = 32
    epochs = 20
    
    training_generator = DatasetGenerator(X_train, Y_train, train_len=train_len, batch_size=batch_size)
    test_generator = DatasetGenerator(X_test, Y_test, train_len=train_len, batch_size=batch_size)
    # make the model
    model = keras.models.Sequential([
        Dense(100, activation='sigmoid', input_shape=[train_len,]),
        Dense(100, activation='sigmoid'),
        Dense(100, activation='sigmoid'),
        Dense(1, activation=None)
    ])
    
    opt = keras.optimizers.Adam(
        learning_rate = 0.001,
        beta_1 = 0.9,
        beta_2 = 0.999
    )
    
    model.compile(
        optimizer=opt,
        loss='mse'
    )
    # train the model
    model.fit(
        training_generator,
        batch_size=None,
        epochs=epochs,
        validation_data=test_generator,
    )
    model.save('./model_saves/pure_nn')
    ## evaluate model
    k_pred_tot = np.zeros((20, k_test.shape[1]-train_len+1))
    for i in range(20):
        k_pred = model.predict(sliding_window_view(X_test[i], [train_len]))
        # undo scaling
        k_pred = k_pred*k_std + k_m
        k_pred_tot[i] = k_pred.flatten()
    
    np.save('./model_predictions/pure_nn/k_pred.npy', k_pred_tot)
    k_true = k_test[:,train_len-1:]
    
    mse = np.mean(np.square(k_pred_tot - k_true))
    rmse = np.sqrt(mse)
    
    print('MSE:', mse)
    print('RMSE:', rmse)
    #%%

if __name__ == '__main__':
    main()