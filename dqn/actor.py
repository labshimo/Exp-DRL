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

class Actor:
    def __init__(self, epsilon=1.0,delta_epsilon=1.0/5000, num_actions=3):
        self.epsilon       = epsilon
        self.delta_epsilon = delta_epsilon
        self.actions       = np.arange(num_actions)  #　PA
        self.n_actions     = num_actions

    def get_action(self,state,targetQN,train=True):      
        q = targetQN.model.predict(state)
        if train and np.random.random()<self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(q[0])

        return action, q[0], np.max(q)
    def reduce_epsilon(self):
        self.epsilon -= self.delta_epsilon


