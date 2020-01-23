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
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling2D ,MaxPooling1D, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras import backend as K
import tensorflow as tf 
from .qrnn import QRNN

class QRCNN1D:
    def __init__(self,current_id,learning_rate=0.01, state_size=4, action_size=2, hidden_size=64):
        self.state_size        = state_size
        self.action_size       = action_size
        self.model             = Sequential()
        self.model.add(Conv1D(self.state_size,10,padding='same',activation='relu',input_shape=(self.state_size,1)))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(QRNN(self.state_size, window_size=3,activation='relu'))
        self.model.add(Dense(action_size, activation ='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam

        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
        self.id = current_id
    def replay(self,memory,batch_num,batch_len,gamma,targetQN):
        state_minibatch = np.zeros((batch_len,batch_num,self.state_size,1))
        y_minibatch = np.zeros((batch_len,batch_num,self.action_size))
        state, state_dash, rewards, actions = memory.sample(batch_num, batch_len)
        transpose_index = np.arange(len(state.shape))
        transpose_index[0] = 1
        transpose_index[1] = 0

        state = state.transpose(*transpose_index)
        state_dash = state_dash.transpose(*transpose_index)

        self.model.reset_states()       # Refresh model_main's state
        targetQN.model.reset_states() # Refresh model_target's state
        s0 = state[0]
        targetQN.model.predict(s0.reshape(-1,self.state_size,1)) # Update target model initial state
        
        for i in range(batch_len):
            #[ seq..., action, reward, seq_new]
            s        = state[i]
            sd       = state_dash[i]
            a        = actions[i].astype(np.int)
            r        = rewards[i]
            q        = self.model.predict(s.reshape(-1,self.state_size,1))
            # Q(s',*): shape is (batch_size, action_num)
            q_dash = targetQN.model.predict(sd.reshape(-1,self.state_size,1))
            max_q_dash = q_dash.max(axis=1) 
            # update Q value
            for (xi, q_) in enumerate(q):
                q_[a[xi]]=r[xi] + gamma * max_q_dash[xi]
            state_minibatch[i,:,:]=s.reshape(-1,self.state_size,1)
            y_minibatch[i,:]=q
        
        state_minibatch = state_minibatch.reshape(batch_len*batch_num,self.state_size,1)              
        y_minibatch = y_minibatch.reshape(batch_len*batch_num,self.action_size)              
        self.model.fit(state_minibatch, y_minibatch,batch_size=batch_len*batch_num,verbose=0)
        loss = self.model.evaluate(state_minibatch, y_minibatch,batch_size=batch_len*batch_num,verbose=0)
        return loss
    def load_model(self, name_y, name_w):
        f_model= 'C:/Users/flabexp/Documents/DQN/Experiment/data/January13212234/trained_model'
        #f_model     = '../data/'+self.id+'/trained_model'
        print('load model')
        json_string = open(os.path.join(f_model, name_y)).read()
        self.model  = model_from_json(json_string)
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)
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