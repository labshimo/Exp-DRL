# coding: utf-8
import numpy as np
import time
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras import backend as K
import tensorflow as tf
from scipy.ndimage.interpolation import shift

class Memory:
    def __init__(self, state_num, max_size=1000, max_size_local=300, position_num=2):
        self.state_num              = state_num
        self.max_size_local         = max_size_local
        self.memSize                = max_size
        self.i                      = 0
        self.memPos                 = 0 
        self.position_num           = position_num
        self.sep                    = -0.15
        self.calib                  = -0.54
        self.start                  = 100
        self.end                    = 200
        self.batch_size_local       = self.end - self.start 
        self.global_memsize         = int(self.memSize/self.batch_size_local)
        self.total_reward_award     = np.ones(int(self.memSize/self.batch_size_local))*-1000
        # initialize local memory
        self.initialize_local_memory() 
        # initialize global memory
        self.episodes               = np.zeros((0,self.state_num * 2 + 4,self.position_num))
    def calc_calibulation(self):
        self.calib=np.average(self.cpte[:self.start])

    def initialize_local_memory(self):
        self.state                  = np.zeros((self.max_size_local,self.state_num,self.position_num))
        self.action                 = np.zeros(self.max_size_local)
        self.next_state             = np.zeros((self.max_size_local,self.state_num,self.position_num))
        self.cpte                   = np.zeros(self.max_size_local)
        self.reward                 = np.zeros(self.max_size_local)
        self.punish                 = np.zeros(self.max_size_local)
        self.episode_local          = np.zeros((self.max_size_local,self.state_num * 2 + 4,self.position_num))
        self.size                   = 0

    def add_local(self,state_t,action_t,next_state_t,cpte_t,reward_t,punish_t):
        # episode local memory
        self.state[self.size]      = state_t
        self.action[self.size]     = action_t
        self.next_state[self.size] = next_state_t
        self.punish[self.size]     = punish_t
        self.cpte[self.size]       = cpte_t
        self.reward[self.size]     = reward_t
        self.size +=1

    def add_global(self,total_reward):
        # global memory # keep best 1-100 
        if np.min(self.total_reward_award)<=total_reward:
            self.i = np.argmin(self.total_reward_award)
            self.memPos = self.i*int(self.end - self.start)
            self.total_reward_award[self.i] = total_reward
            # GOOD EXPERIENCE REPLAY
            for x in self.episode_local[self.start:self.end]:
                self.experience( x )

        # extract random memory
        elif np.random.random()<0.5:
            self.i = np.random.randint(0,self.global_memsize)
            self.memPos = self.i * int(self.end - self.start)
            self.total_reward_award[self.i] = total_reward
            # # NORMAL EXPERIENCE REPLAY
            for x in self.episode_local[self.start:self.end]:
                self.experience( x )

        # initialize episode local memory
        self.size = 0
        # initialize memory
        self.initialize_local_memory()

    def experience(self,x):
        if len(self.episodes)>=self.memSize:
            self.episodes[int(self.memPos%self.memSize)] = x
            self.memPos += 1
        else:
            #self.experienceMemory.extend( x )
            self.episodes = np.concatenate((self.episodes, x.reshape(1,self.state_num * 2 + 4,self.position_num)),axis = 0)

    def edit_experience_local(self):

        self.episode_local[:,0:self.state_num,:]                    = self.state
        self.episode_local[:,self.state_num,0]                      = self.action
        self.episode_local[:,self.state_num+1,0]                    = self.reward
        self.episode_local[:,self.state_num+2:self.state_num*2+2,:] = self.next_state
        self.episode_local[:,self.state_num*2+2,0]                  = self.cpte
        self.episode_local[:,self.state_num*2+3,0]                  = self.punish
    
    def totalreward(self):
        total = np.sum(self.episode_local[self.start:self.end,:,0], axis = 0)
        return total[self.state_num+1]

    def sample(self, batch_num):
        size=len(self.episodes)
        batch_index = list(np.random.randint(0,size,(batch_num)))
        return np.array( [self.episodes[i] for i in batch_index ])

    def len(self):
        return len(self.episodes)
