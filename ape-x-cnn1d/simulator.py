# coding: utf-8
import numpy as np
from numpy import pi
import time
import os
from collections import deque
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
    def __init__(self,args,Property):
        self.property                                  = Property
        self.dynamic_pressre                           = 57
        self.reward_indicator                          = -0.55
        self.sample_rate, self.number_of_samples       = self.property.rate()
        self.input_channel, self.sens_coff             = self.property.i_channels()
        self.output_channel                            = self.property.output()
        self.state_channel, self.loc_100               = self.property.s_channels()
        self.total_time, self.n_loop                   = self.property.control()
        self.burst_wave_file_name                      = self.property.get_burst_wave_file()
        self.num_i                                     = len(self.sens_coff)
        self.num_s                                     = len(self.state_channel) 
        self.frame_width                               = args.frame_width
        self.frame_height                              = args.frame_height
        self.state_length                              = args.state_length
        self.num_actions                               = args.n_actions
        # 10 moving averege
        self.num_mave                                  = 10 
        self.b                                         = np.ones(self.num_mave)/self.num_mave
        self.burst_wave                                = self.get_burst_wave()
        self.before_reward                             = 0 
        self.attachment_count                          = 0
    def get_burst_wave(self):
        PA = np.zeros((self.num_actions,self.number_of_samples))
        with open(self.burst_wave_file_name, 'r') as f:
            reader = csv.reader(f)
            for i,row in enumerate(reader):
                PA[i,:] = row

        return PA

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
        self.write_task.out_stream.auto_start = False

        self.writer = AnalogSingleChannelWriter(self.write_task.out_stream)
        self.reader = AnalogMultiChannelReader(self.read_task.in_stream)

    def start_reading(self):
        # start read analog 
        self.read_task.start()
        time.sleep(0.1)
        self.sample_clk_task.start()
        
    def stop_DAQmx(self):
        self.read_task.close()
        self.write_task.close()
        self.sample_clk_task.close()
    
    def get_observation(self):
        values_read = np.zeros((self.num_i,self.number_of_samples), dtype=np.float64)
        self.reader.read_many_sample(values_read, number_of_samples_per_channel=self.number_of_samples,timeout=2)
        return (((values_read / self.sens_coff) + self.dynamic_pressre ) / self.dynamic_pressre).T
    
    def get_initial_state(self):       
        return np.zeros((self.state_length*self.frame_height,self.frame_width))

    def preprocess(self, observation):

        state_t = np.array([ np.convolve(observation[:,self.state_channel[i]], self.b, mode='same') for i in range(self.num_s) ]).T

        return np.round((np.average(np.split(state_t,self.frame_height,0),axis=1)),2).reshape(self.frame_height,self.frame_width)
    
    def write_daqmx_zero(self):
        values_zero = np.zeros(self.number_of_samples)
        self.writer.write_many_sample(values_zero)

        
    def write_daqmx(self,action):
        self.writer.write_many_sample(self.burst_wave[action]*5)

    def get_reward(self, reward_ori):
        reward = self.reward_indicator < reward_ori
        return reward.astype(np.int)

    def get_reward_with_punish(self, reward_ori, action):
        reward = self.reward_indicator < reward_ori
        
        if reward:
            return reward.astype(np.int)*0.7-(action-1)*0.3
        else:
            return reward.astype(np.int)*0.7

    def get_reward_with_keep_attachment(self, reward_ori, action):
        reward = self.reward_indicator < reward_ori
       
        if (reward.astype(np.int)-self.before_reward) >= 0 and reward:
            self.attachment_count += 0.01
        else:
            self.attachment_count = 0.0

        if reward:
            return self.attachment_count-(action-1)*0.1
        else:
            return self.attachment_count
