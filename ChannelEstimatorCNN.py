
from keras.layers import Conv2D, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import numpy as np
from copy import copy
from scipy import misc
import pickle
import math
from scipy import io

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 20
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

class ChannelEstimatorCNN():
    def __init__(self, input_size = (72, 14), noise_std=0.05):
        self.model = self.getModel(input_size)
        self.noise_std = noise_std
        self.input_size = input_size
        self.scaler = {}
        self.scaler['min'] = 0
        self.scaler['max'] = 0


    def getModel(self, input_size):
        initializer = 'zeros' #'he_normal'
        x = Input(shape = input_size+(2, ))
        c1 = Conv2D(64, (6, 4), activation = 'relu', kernel_initializer = initializer , padding='same')(x)
        c2 = Conv2D(32, (1, 1), activation = 'relu', kernel_initializer = initializer, padding='same')(c1)
        c3 = Conv2D(2, (3, 2), kernel_initializer = initializer, padding='same')(c2)
        model = Model(x, c3)
        model.summary()
        ##compile
        adam = Adam(lr=0.0001)#, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
        model.compile(loss='mse', optimizer=adam)  
        return model

    def train(self, channels, batch_size=64, epochs=100, ):
        copy_channel = copy(channels)
        sampled_channels = copy_channel[:,1::6, 1::4,:]
        noisy_sampled_channels = sampled_channels + self.noise_std * np.random.randn(*sampled_channels.shape)
        cubic_estimated_channels = np.zeros(channels.shape)
        for i,im in enumerate(noisy_sampled_channels):
            cubic_estimated_channels[i,:,:,0] = misc.imresize(im[:,:,0], size=self.input_size, mode='F')
            cubic_estimated_channels[i,:,:,1] = misc.imresize(im[:,:,1], size=self.input_size, mode='F')
        
        # scaling
        self.scaler['max'] = np.max(cubic_estimated_channels)
        self.scaler['min'] = np.min(cubic_estimated_channels)
        noisy_channels_scaled = (cubic_estimated_channels - self.scaler['min']) / (self.scaler['max'] - self.scaler['min'])
        channels_scaled = (channels - self.scaler['min']) / (self.scaler['max'] - self.scaler['min'])
        lrate = LearningRateScheduler(step_decay)
        history = self.model.fit(noisy_channels_scaled, channels_scaled, batch_size=batch_size, epochs=epochs, callbacks = [],
          verbose=1,validation_split=0.15)
        return (noisy_channels_scaled[0], channels_scaled[0])
    
    def test(self, channels):
        copy_channels = copy(channels)
        sampled_channels = copy_channels[:,1::6, 1::4,:]
        cubic_estimated_channels = np.zeros(channels.shape)
        for i,im in enumerate(sampled_channels):
            cubic_estimated_channels[i,:,:,0] = misc.imresize(im[:,:,0], size=self.input_size, mode='F')
            cubic_estimated_channels[i,:,:,1] = misc.imresize(im[:,:,1], size=self.input_size, mode='F')
        
        noisy_channels_scaled = (cubic_estimated_channels - self.scaler['min']) / (self.scaler['max'] - self.scaler['min'])

        predicted_channels_scaled = self.model.predict(noisy_channels_scaled)
        predicted_channels = predicted_channels_scaled*(self.scaler['max'] - self.scaler['min']) + self.scaler['min']
        return predicted_channels

    def saveModel(self, modelName = 'cnn'):
        self.model.save(modelName+'_model.h5')
        with open(modelName+'_scaler.pkl', 'wb') as handle:
            pickle.dump(self.scaler, handle)

    def loadModel(self, modelName = 'cnn'):
        self.model.load_weights(modelName+'_model.h5')
        with open(modelName+'_scaler.pkl', 'rb') as handle:       
            self.scaler = pickle.load(handle)

        

if __name__ == '__main__':
    Data_file = 'Ch_real_VehA_14.mat'
    model = ChannelEstimatorCNN()
    
    ##load train data
    channels = io.loadmat(Data_file)['channels']
    reals = np.real(channels)
    imags = np.imag(channels)

    # desire outputs
    out_train = np.zeros(reals.shape+(2,))
    out_train[:,:,:,0] = reals
    out_train[:,:,:,1] = imags

    model.train(out_train)