# coding: utf-8
from __future__ import division, print_function
from numpy import pi
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras import backend as K
import tensorflow as tf 
import csv
from scipy.ndimage.interpolation import shift 
import time
import pandas as pd

import nidaqmx
from nidaqmx.constants import (Edge, TriggerType, AcquisitionType, LineGrouping, Level, TaskMode, RegenerationMode)
from nidaqmx.utils import flatten_channel_string
from nidaqmx.stream_readers import (AnalogSingleChannelReader, AnalogMultiChannelReader)
from nidaqmx.stream_writers import (AnalogSingleChannelWriter, AnalogMultiChannelWriter)

class Simulator():
    def __init__(self,statenum,num_actions,Actor,QN,memory,gamma,batch_num,sample_rate,number_of_samples,
        input_channel,state_channel,loc_100,sensor_coff,output_channel,nloops,PA,current_id):

        self.actor              = Actor
        self.num_actions        = num_actions
        self.QN                 = QN
        self.memory             = memory
        self.gamma              = gamma
        self.batch_num          = batch_num
        self.sample_rate        = sample_rate
        self.number_of_samples  = number_of_samples
        self.input_channel      = input_channel # input cahnnels
        self.state_channel      = state_channel
        self.output_channel     = output_channel
        self.number_of_channels = sensor_coff.shape[0]
        self.state_length       = statenum
        self.state_width        = len(state_channel) # number of input cahnnels
        self.CH100              = loc_100
        self.nloops             = nloops
        self.burst_wave         = PA
        self.loss               = 0
        self.dynamic_pressre    = 60
        self.sensor_coff        = sensor_coff
        self.intispn            = 0.1 # input time span
        self.tilen              = int(self.intispn / 0.01)
        self.state_height       = int(self.state_length/self.tilen)
        self.id                 = current_id
        # 10 moving averege
        self.num_mave           = 10 
        self.b                  = np.ones(self.num_mave)/self.num_mave
        # inital action of DNN
        self.init_model_action()

        self.total_reward        = 0
        self.total_q_max         = 0
        self.reward_indicator    = -0.55
        self.a_buffer            = [0] * self.num_actions

    def init_model_action(self):
        # first call of keras needs 0.22s
        seq = np.zeros(self.state_length*self.state_width).reshape(-1,self.state_length,self.state_width)
        #seq = np.zeros(self.state_length*self.state_width).reshape(-1,self.state_length)
        # keras needs to drive in 0.001s
        for _ in range(5):
            start = time.time()
            self.actor.get_action(seq,self.QN,train=False)
            end = time.time()
            print(start - end)

    def save_csv_3Dto2D(self,filename,data3d):
        #get absolute path 
        name = os.path.dirname(os.path.abspath(__name__)) 
        #conbine absolute and relative path
        joined_path = os.path.join(name, '../data/', self.id, filename)
        # change to absolute path 
        data_path = os.path.normpath(joined_path)

        data2d = np.zeros((self.nloops*3,self.state_length*self.state_width))
        for i in range(len(data3d)):
            data2d[i] = data3d[i].T.reshape(self.state_length*self.state_width)


        df = pd.DataFrame(data2d)
        df.to_csv(data_path)

    def save_csv(self,filename,data):
        #get absolute path 
        name = os.path.dirname(os.path.abspath(__name__)) 
        #conbine absolute and relative path
        joined_path = os.path.join(name, '../data/', self.id, filename)
        # change to absolute path 
        data_path = os.path.normpath(joined_path)
        df = pd.DataFrame(data)
        df.to_csv(data_path)

    def save(self,read,i):
        #save data to csv file
        localdata = "runs/state/s%d.csv"%(i)
        self.save_csv_3Dto2D(localdata,self.memory.episode_local[:,0:self.state_length,:])
        localdata = "runs/state_/s_%d.csv"%(i)
        self.save_csv_3Dto2D(localdata,self.memory.episode_local[:,self.state_length+2:self.state_length*2+2,:])
        
        localdata = "runs/action/a%d.csv"%(i)
        self.save_csv(localdata,self.memory.episode_local[:,self.state_length,0])
        localdata = "runs/reward/r%d.csv"%(i)
        self.save_csv(localdata,self.memory.episode_local[:,self.state_length+1,0])
        localdata = "runs/cpte/cp%d.csv"%(i)
        self.save_csv(localdata,self.memory.episode_local[:,self.state_length*2+2,0])
        localdata = "runs/punish/p%d.csv"%(i)
        self.save_csv(localdata,self.memory.episode_local[:,self.state_length*2+3,0])

        localdata = "runs/result/run%d.csv"%(i)
        self.save_csv(localdata,self.buffer_memory)

        readdata  = "origin/data%d.csv"%(i)
        self.save_csv(readdata,read)

    def setup_DAQmx(self):
        self.read_task         = nidaqmx.Task() 
        self.write_task        = nidaqmx.Task() 
        self.sample_clk_task   = nidaqmx.Task()
        # Use a counter output pulse train task as the sample clock source
        # for both the AI and AO tasks.
        self.sample_clk_task.co_channels.add_co_pulse_chan_freq('Dev1/ctr0', freq=self.sample_rate, idle_state=Level.LOW)
        self.sample_clk_task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS,samps_per_chan=self.number_of_samples)
        self.sample_clk_task.control(TaskMode.TASK_COMMIT)
        samp_clk_terminal = '/Dev1/Ctr0InternalOutput'

        self.read_task.ai_channels.add_ai_voltage_chan(self.input_channel, max_val=10, min_val=-10)
        self.read_task.timing.cfg_samp_clk_timing(self.sample_rate, source=samp_clk_terminal, active_edge=Edge.FALLING,sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=self.number_of_samples)

        self.write_task.ao_channels.add_ao_voltage_chan(self.output_channel, max_val=10, min_val=-10)
        self.write_task.timing.cfg_samp_clk_timing(self.sample_rate, source=samp_clk_terminal, active_edge=Edge.FALLING,sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=self.number_of_samples)
        self.write_task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION
        self.write_task.out_stream.auto_start = True

        self.writer = AnalogSingleChannelWriter(self.write_task.out_stream)
        self.reader = AnalogMultiChannelReader(self.read_task.in_stream)

    def start_reading(self):
        # start read analog 
        self.read_task.start()
        time.sleep(0.1)
        self.sample_clk_task.start()
        

    def read_daqmx(self):
        values_read = np.zeros((self.number_of_channels,self.number_of_samples), dtype=np.float64)
        self.reader.read_many_sample(values_read, number_of_samples_per_channel=self.number_of_samples,timeout=2)
        return (((values_read.T / self.sensor_coff) + self.dynamic_pressre ) / self.dynamic_pressre)
    
    def write_daqmx_zero(self):
        values_zero = np.zeros(self.number_of_samples)
        self.writer.write_many_sample(values_zero)
        
    def write_daqmx(self,action):
        self.writer.write_many_sample(self.burst_wave[action]*5)

    def stop_DAQmx(self):
        self.read_task.close()
        self.write_task.close()
        self.sample_clk_task.close()
    
    def preprocess(self, observation):
        # moving average filter
        state_t  = np.zeros((self.number_of_samples, self.state_width))
        
        for ich in range(self.state_width):
            state_t[:,ich] = np.convolve(observation[:,self.state_channel[ich]], self.b, mode='same')
        processed_observation = np.round((np.average(np.split(state_t,self.state_height,0),axis=1)),2)
        return processed_observation.reshape(self.state_height,self.state_width)

    def get_initial_state(self):       
        return np.zeros((self.state_length,self.state_width))

    def get_reward(self, reward_ori):
        reward = self.reward_indicator < reward_ori
        return reward.astype(np.int)

    def run(self, Nepisode, targetQN, train=True):
        self.setup_DAQmx()
        # Parameters for plotting
        # initiate
        read               = np.zeros((0,self.number_of_channels)) 
        state              = self.get_initial_state()
        self.buffer_memory = np.zeros((0,4+self.num_actions))
        self.total_q_max   = 0
        # start read analog 
        self.start_reading()
        # first loop
        # you read 
        # you do not act 
        # this loop is for caliburation of reward
        # you need to satrt writeanalog in first loop
        # 'cause the function takes time to start 
        self.write_daqmx_zero()
        self.write_task.start()
        self.write_daqmx_zero()
        self.write_daqmx_zero()
        
    
        for n in range(int(self.nloops)):    
            
            # adopt output timing and action zero
            self.write_daqmx_zero()
            observation           = self.read_daqmx()
            processed_observation = self.preprocess(observation)
            next_state            = np.append(state[self.state_height:, :], processed_observation, axis=0)
            # reward
            cpte                  = np.average(observation[:,self.CH100])
            reward                = self.get_reward(cpte)
            read                  = np.append(read,observation,axis=0)
            memory                = [0, cpte, 0, 0]
            memory.extend(self.a_buffer)
            self.memory.add_local(state,0,next_state,cpte,0,0)
            self.buffer_memory    = np.append(self.buffer_memory,[memory],axis=0)
            state                 = next_state
        # calibulatkoe
        self.memory.calc_calibulation()
        # second loop
        # you read 
        # you act
        for n in range(self.nloops):
            # action
            action, q, q_max = self.actor.get_action(state.reshape(-1,self.state_length,self.state_width), self.QN,train)
            #action, q, q_max = self.actor.get_action(state.reshape(-1,self.state_length), self.QN,train)

            # adopt output timing and action zero
            self.write_daqmx(action)
            observation           = self.read_daqmx()
            processed_observation = self.preprocess(observation)
            next_state            = np.append(state[self.state_height:, :], processed_observation, axis=0)
            # reward
            cpte                  = np.average(observation[:,self.CH100])
            reward                = self.get_reward(cpte)
            read                  = np.append(read,observation,axis=0)
            memory                = [action, cpte, reward, np.argmax(q)]
            memory.extend(q)
            self.memory.add_local(state,action,next_state,cpte,reward,0)
            self.buffer_memory    = np.append(self.buffer_memory,[memory],axis=0)
            self.total_q_max     += q_max
            state                 = next_state
        # third loop
        # you read 
        # you do not act
        # make sure PA turn off
        for n in range(int(self.nloops)):
            # adopt output timing and action zero
            self.write_daqmx_zero()
            self.write_daqmx_zero()

            observation           = self.read_daqmx()
            processed_observation = self.preprocess(observation)
            next_state            = np.append(state[self.state_height:, :], processed_observation, axis=0)
            # reward
            cpte                  = np.average(observation[:,self.CH100])
            reward                = self.get_reward(cpte)
            read                  = np.append(read,observation,axis=0)
            memory                = [0, cpte, reward, 0]
            memory.extend(self.a_buffer)
            self.memory.add_local(state,0,next_state,cpte,0,0)
            self.buffer_memory    = np.append(self.buffer_memory,[memory],axis=0)
            state                 = next_state
        # stop DAQmx
        self.stop_DAQmx()
        # edit experience in buffer
        self.memory.edit_experience_local()
        self.save(read,Nepisode)
        self.total_reward = self.memory.totalreward()
        # move current experience to global buffer
        self.memory.add_global(self.total_reward)

        if (self.memory.len() > self.batch_num) and train:
            self.loss=self.QN.replay(self.memory,self.batch_num,self.gamma,targetQN)
            self.actor.reduce_epsilon()

        return self.total_reward, self.loss, self.total_q_max/self.nloops