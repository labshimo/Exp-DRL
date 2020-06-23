# coding: utf-8
import numpy as np
from numpy import pi
import os
from collections import deque
import csv
from scipy.ndimage.interpolation import shift 
import time
import pandas as pd
import json
import nidaqmx
from nidaqmx.constants import (Edge, TriggerType, AcquisitionType, LineGrouping, Level, TaskMode, RegenerationMode)
from nidaqmx.utils import flatten_channel_string
from nidaqmx.stream_readers import (AnalogSingleChannelReader, AnalogMultiChannelReader)
from nidaqmx.stream_writers import (AnalogSingleChannelWriter, AnalogMultiChannelWriter)
from scipy import signal
class ExpFlowSeparation():
    def __init__(self, simlator_args):
        self.get_setting(simlator_args)

    def get_setting(self, arg):
        # analog input
        self.sample_rate       = arg["sample_rate"]
        self.number_of_samples = arg["number_of_samples"] 
        self.dynamic_pressre   = arg["dynamic_pressure"]
        self.reward_indicator  = arg["reward_indicator"]
        self.input_channel     = arg["input_channels"] 
        sens_coff              = np.array(arg["sens_cofficients"])/arg["unit_change"]*arg["gain"]
        self.sens_coff         = sens_coff.reshape(sens_coff.shape[0],1)
        self.num_i             = len(self.sens_coff)
        # state channels for nueral network
        self.state_channel     = arg["state_channels"] 
        self.num_s             = len(self.state_channel) 
        self.loc_100           = arg["reward_channel"] 
        # analog output
        self.output_channel    = arg["output_channel"] 
        # another parameters
        self.timeout           = arg["timeout"] 
        self.dt                = 1/self.sample_rate*self.number_of_samples
        self.total_time        = arg["total_time"] 
        self.n_loop            = int(self.sample_rate*self.total_time/self.number_of_samples)
        # plama actuator 
        self.burst_wave        = self.get_burst_wave(arg)
        self.nb_actions        = self.burst_wave.shape[0] - 1 

    def get_burst_wave(self,arg):
        if arg["mode"]=="gate":
            wave = self.get_gate_wave(arg["gate_mode"]["plasma_actuator_csv"])                        
        elif arg["mode"]=="sin":
            wave = self.create_burst_wave(**arg["sin_mode"])
        print("MODE: " + arg["mode"])
        zero_wave  = np.zeros((1,self.number_of_samples))
        wave       = np.append(zero_wave,wave,axis=0)
        return wave

    def get_gate_wave(self, filename):
        df_wave    = pd.read_csv(filename, header=None).values
        np_wave    = np.array([ i_wave for i_wave in df_wave ])
        return np_wave

    def create_burst_wave(self,base_frequency, burst_frequency, burst_ratio, voltage):
        ### example -> burst_freq=600[Hz], burst_ratio=0.1[-], voltage=3[kV]
        time            = np.linspace(0.0, self.dt, self.number_of_samples)
        tmp_sin         = np.sin(2*np.pi*int(base_frequency)*time)
        tmp_sq          = [(signal.square(2 * np.pi * bf_i * time, duty=br_i)+1)/2 for br_i in burst_ratio for bf_i in burst_frequency]
        wave            = [(tmp_sin * tmp_sq_i) * vi for tmp_sq_i in tmp_sq for vi in voltage] 
        zero_wave       = np.zeros((1,self.number_of_samples)) 
        wave            = np.append(wave,zero_wave,axis=0)# plus off at last array
        return wave

    def load_args(self, filename):
        with open(filename,"r") as f:
            args = json.load(f) 
        return args

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

    def reset(self):
        self.env_memory    = np.zeros((0,self.num_i))
        self.buffer_memory = np.zeros((0,4+self.nb_actions))
        self.step_count = 0
        self.setup_DAQmx()
        # start read analog 
        self._start_reading()
        self._start_writing()
        return self._reading()

    def step(self, action):
        self._writing(action)
        observation = self._reading() 
        reward = np.average(observation[:,self.loc_100])
        if action != self.nb_actions :
            self.step_count += 1
            
        if self.step_count < self.n_loop: 
            return observation, reward, False
        else:
            return observation, reward, True

    def _start_reading(self):
        # start read analog 
        self.read_task.start()
        self.sample_clk_task.start()

    def _start_writing(self):
        self._writing(0)   
        self.write_task.start()
        for _ in range(3):
            self._writing(0)     
        
    def stop_DAQmx(self):
        self.read_task.close()
        self.write_task.close()
        self.sample_clk_task.close()
    
    def _reading(self):
        values_read = np.zeros((self.num_i,self.number_of_samples), dtype=np.float64)
        self.reader.read_many_sample(values_read, number_of_samples_per_channel=self.number_of_samples,timeout=2)
        return (((values_read / self.sens_coff) + self.dynamic_pressre ) / self.dynamic_pressre).T
        
    def _writing(self,action):
        self.writer.write_many_sample(self.burst_wave[action])
    
    