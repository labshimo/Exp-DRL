# coding: utf-8
from __future__ import division, print_function
from numpy import pi
import numpy as np
from numpy import random  
import time
from collections import deque
import os
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Conv1D
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras import backend as K
import tensorflow as tf 

class DQN:
    def __init__(self,current_id,learning_rate=0.01, state_size=4, action_size=2, hidden_size=64):
        self.state_size        = state_size
        self.action_size       = action_size
        self.model             = Sequential()
        self.model.add(Dense(state_size,activation='relu',input_shape=(1,state_size)))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation ='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        self.id = current_id
    def replay(self,memory,batch_num,gamma,targetQN):
        state_minibatch = np.zeros((batch_num,self.state_size))
        y_minibatch = np.zeros((batch_num,self.action_size))
        batch = memory.sample(batch_num)

        for i in range(batch_num):
            #[ seq..., action, reward, seq_new]
            s_j=  batch[i,0:self.state_size]
            a_j = int(batch[i,self.state_size])
            r_j = batch[i, self.state_size+1]
            s_dash_j= batch[i,(self.state_size+2):(self.state_size*2+2)].reshape(-1,1,self.state_size)
            y_j = self.model.predict(s_j.reshape(-1, 1,self.state_size))[0,0]
            y_j[a_j]=( r_j+ gamma * np.max(targetQN.model.predict(s_dash_j)))
            
            state_minibatch[i,:]=s_j
            y_minibatch[i,:]=y_j
        state_minibatch=state_minibatch.reshape(batch_num,1,self.state_size)
        y_minibatch=y_minibatch.reshape(batch_num,1,self.action_size)
        self.model.fit(state_minibatch, y_minibatch,batch_size=int(batch_num/10),epochs=10,verbose=0)
        loss = self.model.evaluate(state_minibatch, y_minibatch,batch_size=int(batch_num/10),verbose=0)
        return loss
    def load_model(self, name_y, name_w):
        f_model     = '../data/'+self.id+'/trained_model'
        print('load model')
        json_string = open(os.path.join(f_model, name_y)).read()
        self.model  = model_from_json(json_string)
        self.model.load_weights(os.path.join(f_model, name_w))
    def save_model(self,num_episode):
        f_model     = '../data/'+self.id+'/trained_model'
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