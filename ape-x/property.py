# coding: utf-8
import daqmx
import numpy as np
import pandas as pd

class Property():
    def __init__(self):
        # analog input
        self.input_frequency     = 20000.0 # freaquency
        self.input_buffer        = 200  # buffer
        self.input_samps_size    = 200 # buffer 
        self.phys_chan_I1        = "Dev1/ai0"
        self.phys_chan_I2        = "Dev1/ai1"
        self.phys_chan_I3        = "Dev1/ai2"
        self.phys_chan_I4        = "Dev1/ai3"
        self.phys_chan_I5        = "Dev1/ai4"
        self.phys_chan_I6        = "Dev1/ai5"
        self.phys_chan_I7        = "Dev1/ai6"
        self.phys_chan_I8        = "Dev1/ai7"
        self.phys_chan_I9        = "Dev1/ai8"
        self.phys_chan_I10       = "Dev1/ai9"
        self.input_channels      = [self.phys_chan_I1,self.phys_chan_I2,self.phys_chan_I3,self.phys_chan_I4,self.phys_chan_I5]
        self.sens_cofficients    = np.array([22.984,23.261,22.850,35.801,25.222])/1000000*900
        # state channels for nueral network
        self.state_channels      = [0,1,2,3,4]
        self.loc_num_100         = 4
        # analog output
        self.output_frequency    = 12000.0 # frequency
        self.output_buffer       = 120 # buffer
        self.phy_chan_O          = "Dev1/ao0"
        # another parameters
        self.mode                = daqmx.Val_GroupByChannel
        self.timeout             = 10
        self.dt                  = 1/self.input_frequency
        self.total_time          = 1
        self.max_number_of_steps = int(self.input_frequency*self.total_time/self.input_samps_size)
        # plama actuator 
        self.filename            = "PA120.csv" 

    def input(self):
        print('get input property!')
        return self.input_frequency, self.input_buffer, self.input_samps_size

    def i_channels(self):
        print('get input channels property!')
        return self.input_channels, self.sens_cofficients

    def output(self):
        print('get output property!')
        return self.output_frequency, self.output_buffer, self.phy_chan_O
        
    def s_channels(self):
        print('get state channels property!')
        return self.state_channels, self.loc_num_100

    def daqmx_property(self):
        print('get daqmx property!')
        return self.mode, self.timeout

    def control(self):
        print('get control property!')
        self.max_number_of_steps = int(self.input_frequency*self.total_time/self.input_samps_size)
        return self.total_time, self.max_number_of_steps

    def get_burst_wave(self):
        # import nondimentional frequency
        df = pd.read_csv('PA120.csv',header=None,index_col=None)
        
        return df.values
            
            
            
            
            
            

