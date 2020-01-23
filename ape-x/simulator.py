
# coding: utf-8

# In[ ]:


# coding: utf-8
import daqmx
import numpy as np
from numpy import pi
import time
import os
from collections import deque
import csv
from scipy.ndimage.interpolation import shift 
import time
import pandas as pd

class Simulator():
    def __init__(self,args,Property):
        self.property                                = Property
        self.dynamic_pressre                         = 55
        self.reward_indicator                        = -0.5 
        
        self.i_frequency, self.i_buffer, self.i_size = self.property.input()
        self.i_channels, self.sens_coff              = self.property.i_channels()
        self.o_fre, self.o_buffer, self.o_channel    = self.property.output()
        self.s_channels, self.loc_100                = self.property.s_channels()
        self.mode, self.timeout                      = self.property.daqmx_property()
        self.total_time, self.n_loop                 = self.property.control()
        self.num_i                                   = len(self.i_channels)
        self.num_s                                   = len(self.s_channels) 
        self.frame_width                             = args.frame_width
        self.frame_height                            = args.frame_height
        self.state_length                            = args.state_length
        # 10 moving averege
        self.num_mave                                = 10 
        self.b                                       = np.ones(self.num_mave)/self.num_mave
        self.pa                                      = self.property.get_burst_wave()
    def setup_DAQmx(self):
        # daqmx tasks
        self.AItask = daqmx.TaskHandle()
        self.AOtask = daqmx.TaskHandle()
        daqmx.CreateTask("", self.AItask)
        daqmx.CreateTask("", self.AOtask)
        # AI task setup
        for i in range(self.num_i):
            daqmx.CreateAIVoltageChan(self.AItask, self.i_channels[i], "", daqmx.Val_Diff,
                                  -10.0, 10.0, daqmx.Val_Volts, None) 

        daqmx.CfgSampClkTiming(self.AItask, "",  self.i_frequency, daqmx.Val_Rising, 
                                daqmx.Val_ContSamps, self.i_size)
        # AO task setup
        daqmx.CreateAOVoltageChan(self.AOtask, self.o_channel, "", 
                                -10.0, 10.0, daqmx.Val_Volts, None)
        daqmx.CfgSampClkTiming(self.AOtask, "", self.o_fre, daqmx.Val_Rising, 
                            daqmx.Val_ContSamps, self.o_buffer)
        daqmx.SetWriteRegenMode(self.AOtask,daqmx.Val_DoNotAllowRegen)     
    def start_reading(self):
        # start read analog 
        daqmx.StartTask(self.AItask, fatalerror=False)
        
    def stop_DAQmx(self):
        daqmx.StopTask(self.AOtask)
        daqmx.StopTask(self.AItask)
        daqmx.ClearTask(self.AItask)
        daqmx.ClearTask(self.AOtask)
    
    def get_observation(self):
        read_t, _ = daqmx.ReadAnalogF64(self.AItask,self.i_size,self.timeout,self.mode,self.i_size,self.num_i)
        return read_t / self.sens_coff
    
    def get_initial_state(self):       
        return np.zeros((self.state_length,self.frame_height,self.frame_width))

    def preprocess(self, observation):
        # moving average filter
        state_t  = np.zeros((self.i_buffer, self.frame_width))

        for ich in range(self.num_s):
            state_t[:,ich] = np.convolve(observation[:,self.s_channels[ich]], self.b, mode='same')
        processed_observation = (np.average(np.split(state_t,self.frame_height,0),axis=1) + self.dynamic_pressre ) / self.dynamic_pressre

        return processed_observation.reshape(1,self.frame_height,self.frame_width)
    
    def write_daqmx_zero(self):
        daqmx.WriteAnalogF64(self.AOtask,self.o_buffer,1,self.timeout,daqmx.Val_GroupByChannel,np.zeros(self.o_buffer))
    
    def write_daqmx(self,ai):
        daqmx.WriteAnalogF64(self.AOtask,self.o_buffer,1,self.timeout,daqmx.Val_GroupByChannel,self.pa[ai,:]*3)
    
    def get_reward(self, reward_ori):
        reward = self.reward_indicator < reward_ori
        return reward.astype(np.int)
