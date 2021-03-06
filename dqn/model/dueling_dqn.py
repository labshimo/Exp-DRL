# coding: utf-8
from __future__ import division, print_function
import numpy as np
from numpy import random  
from collections import deque
import os
from keras.models import Sequential, Model, Input
from keras.models import model_from_json
from keras.layers import Dense, Conv1D,MaxPooling1D,Flatten,BatchNormalization,Activation,Lambda,concatenate
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
#from keras.utils.training_utils import multi_gpu_model # add
import time

gpu_count = 2 # add

def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

class Dueling_CNN:
    def __init__(self,current_id,learning_rate=0.01, state_size=80, position_num=2, action_size=2, hidden_size=80):
        self.state_size        = state_size
        self.position_num      = position_num
        self.hidden_size       = hidden_size
        self.action_size       = action_size
        self.model             = Sequential()
        
        # 1st convolutional layer block
        inputlayer = Input(shape=(self.state_size,self.position_num))
        middlelayer = Conv1D(self.state_size,10,padding='same', name='cnn1d1-1')(inputlayer)
        middlelayer = MaxPooling1D(pool_size=2)(middlelayer)
        middlelayer = Activation('relu')(middlelayer)
        middlelayer = Conv1D(self.state_size,10,padding='same', name='cnn1d1-2')(middlelayer)
        middlelayer = MaxPooling1D(pool_size=2)(middlelayer)
        middlelayer = BatchNormalization(name='Batch1')(middlelayer)
        middlelayer = Activation('relu')(middlelayer)
        # full conected layer
        middlelayer = Flatten()(middlelayer)
        v = Dense(self.state_size, activation='relu')(middlelayer)
        v = Dense(1)(v)
        adv = Dense(self.state_size, activation='relu')(middlelayer)
        adv = Dense(self.action_size)(adv)
        y = concatenate([v,adv])
        outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(self.action_size,))(y)


        self.model=Model(input=inputlayer, output=outputlayer)   
        
        #self.model = multi_gpu_model(self.model, gpus=gpu_count) # add       
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
        self.id = current_id
    def replay(self,memory,batch_num,gamma,targetQN):
        state_minibatch = np.zeros((batch_num,self.state_size,self.position_num))
        y_minibatch = np.zeros((batch_num,self.action_size))
        s_batch  = np.zeros((batch_num,self.state_size,self.position_num))
        sd_batch = np.zeros((batch_num,self.state_size,self.position_num))
        y_batch  = np.zeros((batch_num,self.action_size))
        batch    = memory.sample(batch_num)

        for i in range(batch_num):
            #[ seq..., action, reward, seq_new]
            s_j   = batch[i,0:self.state_size,:]
            s_d_j = batch[i,self.state_size+2:self.state_size*2+2,:]
            s_batch[i,:,:]  = s_j
            sd_batch[i,:,:] = s_d_j

        #y_batch = self.model.predict(s_batch.reshape(batch_num,self.state_size))
        y_batch = self.model.predict(s_batch) 
        #y_dash  = targetQN.model.predict(sd_batch.reshape(batch_num,self.state_size))
        y_dash  = targetQN.model.predict(sd_batch)

        for (i,y_i) in enumerate(y_batch):
            a_j      = int(batch[i,self.state_size,0])
            r_j      = batch[i,self.state_size+1,0]
            y_i[a_j] = r_j+ gamma * np.max(y_dash[i])

        history = self.model.fit(s_batch, y_batch,batch_size=batch_num,verbose=0)
        return history.history['loss'][0]
       
    def load_model(self, name_y, name_w):
        f_model     = 'C:/Users/flabexp/Documents/DQN/Experiment/data/October14151109/trained_model'
        print('load model')
        json_string = open(os.path.join(f_model, name_y)).read()
        self.model  = model_from_json(json_string)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        self.model.load_weights(os.path.join(f_model, name_w))

    def save_model(self,num_episode):
        f_model     = self.id+'/trained_model'
        name_j      = 'model%d.json'%num_episode
        name_y      = 'model%d.yaml'%num_episode
        name_w      = 'weights%d.hdf5'%num_episode
        json_string = self.model.to_json()
        yaml_string = self.model.to_yaml()
        print('save the architecture of a model')
        open(os.path.join(f_model,name_j), 'w').write(json_string)
        open(os.path.join(f_model,name_y), 'w').write(yaml_string)
        print('save weights')
        self.model.save_weights(os.path.join(f_model,name_w))