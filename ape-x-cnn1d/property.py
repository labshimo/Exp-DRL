# coding: utf-8
import numpy as np
import pandas as pd

class Property():
    def __init__(self):
        # analog input
        self.sample_rate         = 12000
        self.number_of_samples   = 120
        self.input_channels      = "Dev1/ai0:4"
        #self.sens_cofficients    = np.array([35.801,25.222])/1000000*900 #11160 11172 11125 11120 11211
        self.sens_cofficients    = np.array([22.984,23.261,22.850,35.801,25.222])/1000000*900 #11160 11172 11125 11120 11211
        self.sens_cofficients    = self.sens_cofficients.reshape(self.sens_cofficients.shape[0],1)

        # state channels for nueral network
        self.state_channels      = [0,3,4]
        self.loc_num_100         = 4
        # analog output
        self.phy_chan_O          = "Dev1/ao0"
        # another parameters
        self.timeout             = 2
        self.dt                  = 1/self.sample_rate
        self.total_time          = 1
        self.max_number_of_steps = int(self.sample_rate*self.total_time/self.number_of_samples)
        # plama actuator 
        self.filename            = "csv/"+"PA120-2.csv" 

    def rate(self):
        print('get input property!')
        return self.sample_rate, self.number_of_samples
    def i_channels(self):
        print('get input channels property!')
        return self.input_channels, self.sens_cofficients

    def output(self):
        print('get output property!')
        return self.phy_chan_O
        
    def s_channels(self):
        print('get state channels property!')
        return self.state_channels, self.loc_num_100

    def control(self):
        print('get control property!')
        self.max_number_of_steps = int(self.sample_rate*self.total_time/self.number_of_samples)
        return self.total_time, self.max_number_of_steps

    def get_burst_wave_file(self):
        return self.filename
            
            
            
            
            
            

