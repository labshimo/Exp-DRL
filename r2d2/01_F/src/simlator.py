import pickle
import os
import numpy as np
import random
import time
import math
from flow_separation import ExpFlowSeparation
import pandas as pd

class SeparationControl():
    def __init__(self, args, simlator_args):
        self.exp          = ExpFlowSeparation(simlator_args)
        self.nb_actions   = self.exp.nb_actions
        self.r_indicator  = self.exp.reward_indicator
        self.number_obs   = args["input_shape"][1]
        self.number_state = len(simlator_args["state_channels"])
        # state channels
        self.state_channel = self.exp.state_channel
        # moving averege setting
        self.num_mave          = args["input_shape"][0]
        self.b                 = np.ones(self.num_mave)/self.num_mave
        self.before_reward     = 0 
        self.attachment_count  = 0
        self.a_buffer          = np.zeros(self.nb_actions)
        if not args["test"]:
            self.data_path         = args["log_dir"] + "data/"
            self.run_path          = args["log_dir"] + "run/"
        else:
            self.data_path         = args["test_dir"] + "data/"
            self.run_path          = args["test_dir"] + "run/"

        if not (os.path.exists(self.data_path) or os.path.exists(self.run_path)):
            os.makedirs(self.data_path)
            os.makedirs(self.run_path)


    def reset(self):
        observation_ori = self.exp.reset()
        observation = self.process_observation(observation_ori)
        self.buffer_memory = np.zeros((0,3+self.nb_actions))
        self.env_memory    = observation_ori
        return observation
    
    def step(self, action, q_values):
        observation_ori, reward_ori, terminal = self.exp.step(action)
        observation = self.process_observation(observation_ori)

        reward = self.process_reward(reward_ori)

        data = [action, reward_ori, reward]
        data.extend(q_values)
        self.buffer_memory = np.append(self.buffer_memory,[data],axis=0)
        self.env_memory    = np.append(self.env_memory,observation_ori,axis=0)
        return observation, reward, terminal
        
    def stop(self):
        self.exp.stop_DAQmx()

    def process_observation(self, observation):
        prpcessed_observation = np.array([ np.convolve(observation[:,self.state_channel[i]], self.b, mode='same') for i in range(self.number_state) ]).T
        return np.round((np.average(np.split(prpcessed_observation,self.num_mave,0),axis=1)),2)

    def process_reward(self, reward_ori):
        reward = self.r_indicator < reward_ori
        return reward.astype(np.int)

    def _rewarding(self, reward_ori):
        reward = self.r_indicator < reward_ori
        return reward.astype(np.int)

    def get_reward_with_punish(self, reward_ori, action):
        reward = self.r_indicator < reward_ori
        
        if reward:
            return reward.astype(np.int)*0.7-(action-1)*0.3
        else:
            return reward.astype(np.int)*0.7

    def get_reward_with_keep_attachment(self, reward_ori, action):
        reward = self.r_indicator < reward_ori
       
        if (reward.astype(np.int)-self.before_reward) >= 0 and reward:
            self.attachment_count += 0.01
        else:
            self.attachment_count = 0.0

        if reward:
            return self.attachment_count-(action-1)*0.1
        else:
            return self.attachment_count

    def save_simlator_data(self,episode):
        data_save_path = self.data_path + '/data{:0=5}.csv'.format(episode)
        run_save_path  = self.run_path + '/run{:0=5}.csv'.format(episode)
        df_d = pd.DataFrame(self.env_memory)
        df_r = pd.DataFrame(self.buffer_memory)
        df_d.to_csv(data_save_path)
        df_r.to_csv(run_save_path)