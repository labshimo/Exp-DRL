# coding: utf-8
from __future__ import division, print_function
import daqmx
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

class Simulator():
    def __init__(self,statenum,Actor,QN,memory,gamma,batch_num,AIfre,samps_per_chan,
                array_size_samps,fillmode,phy_channels,input_channels,chnane_num_100,sensor_coff,phy_chan_O,
                AOfre,Aobu,timeout,nloops,PA,current_id):
        self.actor           = Actor
        self.QN              = QN
        self.memory          = memory
        self.gamma           = gamma
        self.batch_num       = batch_num
        self.num             = statenum
        self.IF              = AIfre
        self.SPC             = samps_per_chan
        self.ASS             = array_size_samps
        self.MODE            = fillmode
        self.AIs             = phy_channels
        self.IAIs            = input_channels # input cahnnels
        self.O1              = phy_chan_O
        self.NCH             = len(phy_channels)
        self.NICH            = len(input_channels) # number of input cahnnels
        self.CH100           = chnane_num_100
        self.OF              = AOfre
        self.OB              = Aobu
        self.TO              = timeout
        self.nloops          = nloops
        self.PA              = PA
        self.fc              = 6000
        self.freq            = np.fft.fftfreq(self.SPC)
        self.loss            = 0
        self.dynamic_pressre = 55
        self.sensor_coff     = sensor_coff
        self.intispn         = 0.1 # input time span
        self.id              = current_id
        # 10 moving averege
        self.num_mave        = 10 
        # inital action of DNN
        self.init_model_action()

    def init_model_action(self):
        # first call of keras needs 0.22s
        seq = np.zeros(self.num*self.NICH).reshape(-1,self.num,self.NICH)
        # keras needs to drive in 0.001s
        for _ in range(5):
            s=time.time()
            self.actor.get_action(seq,self.QN,train=False)
            f=time.time()
            print(f-s)

    def save_csv_3Dto2D(self,filename,data3d):
        #get absolute path 
        name = os.path.dirname(os.path.abspath(__name__)) 
        #conbine absolute and relative path
        joined_path = os.path.join(name, '../data/', self.id, filename)
        # change to absolute path 
        data_path = os.path.normpath(joined_path)

        data2d = np.zeros((self.nloops*3,self.num*self.NICH))
        for i in range(len(data3d)):
            data2d[i] = data3d[i].T.reshape(self.num*self.NICH)


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
        self.save_csv_3Dto2D(localdata,self.memory.episode_local[:,0:self.num,:])
        localdata = "runs/state_/s_%d.csv"%(i)
        self.save_csv_3Dto2D(localdata,self.memory.episode_local[:,self.num+2:self.num*2+2,:])
        localdata = "runs/action/a%d.csv"%(i)
        self.save_csv(localdata,self.memory.episode_local[:,self.num,0])
        localdata = "runs/reward/r%d.csv"%(i)
        self.save_csv(localdata,self.memory.episode_local[:,self.num+1,0])
        localdata = "runs/cpte/cp%d.csv"%(i)
        self.save_csv(localdata,self.memory.episode_local[:,self.num*2+2,0])
        localdata = "runs/punish/p%d.csv"%(i)
        self.save_csv(localdata,self.memory.episode_local[:,self.num*2+3,0])
        readdata  = "origin/data%d.csv"%(i)
        self.save_csv(readdata,read)

    def lowpass_filter(self,read):
        F = np.fft.fft(read)/(self.SPC/2)
        F[0] = F[0]/2
        F[(self.freq > self.fc)] = 0
        F[(self.freq < 0)] = 0
        return np.real(np.fft.ifft(F)*(2*self.SPC/2))

    def setup_DAQmx(self):
        # daqmx tasks
        self.AItask = daqmx.TaskHandle()
        self.AOtask = daqmx.TaskHandle()
        daqmx.CreateTask("", self.AItask)
        daqmx.CreateTask("", self.AOtask)
        # AI task setup
        for i in range(self.NCH):
            daqmx.CreateAIVoltageChan(self.AItask, self.AIs[i], "", daqmx.Val_Diff,
                                  -10.0, 10.0, daqmx.Val_Volts, None) 

        daqmx.CfgSampClkTiming(self.AItask, "",  self.IF, daqmx.Val_Rising, 
                                daqmx.Val_ContSamps, self.ASS)
        # AO task setup
        daqmx.CreateAOVoltageChan(self.AOtask, self.O1, "", 
                                -10.0, 10.0, daqmx.Val_Volts, None)
        daqmx.CfgSampClkTiming(self.AOtask, "", self.OF, daqmx.Val_Rising, 
                            daqmx.Val_ContSamps, self.OB)
        daqmx.SetWriteRegenMode(self.AOtask,daqmx.Val_DoNotAllowRegen)     

    def stop_DAQmx(self):
        daqmx.StopTask(self.AOtask)
        daqmx.StopTask(self.AItask)
        daqmx.ClearTask(self.AItask)
        daqmx.ClearTask(self.AOtask)
        
    def run(self, Nepisode, targetQN, train=True):
        self.setup_DAQmx()
        # Parameters for plotting
        # initiate
        seq      = np.zeros((self.num,self.NICH))
        read     = np.zeros((0,self.NCH)) 
        state_t  = np.zeros((self.SPC,self.NICH))
        b        = np.ones(self.num_mave)/self.num_mave
        tilen    = int(self.intispn / 0.01)
        # start read analog 
        daqmx.StartTask(self.AItask, fatalerror=False)
        # first loop
        # you read 
        # you do not act 
        # this loop is for caliburation of reward
        # you need to satrt writeanalog in first loop
        # 'cause the function takes time to start 
        for n in range(self.nloops):    
            read_t, _ = daqmx.ReadAnalogF64(self.AItask,self.SPC,self.TO,self.MODE,self.ASS,self.NCH)
            read_t = read_t/self.sensor_coff
        
            # moving average filter
            for ich in range(self.NICH):
                state_t[self.num_mave-1:,ich] = np.convolve(read_t[:,self.IAIs[ich]], b, mode='vaild')

            state_ave = ( np.average(np.split(state_t,int(self.num/tilen),0),axis=1) + self.dynamic_pressre )/self.dynamic_pressre
            seq[self.num-int(self.num/tilen):self.num,:] = np.round(state_ave,2)

            # action
            # adopt output timing and action zero
            if n!=10 or n!=20 or n!=40 or n!=50 or n!=60 or n!=80 or n!=90:
                daqmx.WriteAnalogF64(self.AOtask,self.OB,1,self.TO,daqmx.Val_GroupByChannel,np.zeros(self.OB))
            # reward
            reward_t  = np.average(read_t[:,self.CH100])
            self.memory.add_local(seq,0,reward_t,0)
            read = np.append(read,read_t,axis=0)
            seq = shift(seq,[-self.num/tilen,0],cval=0)

        # calibulate
        self.memory.calc_calibulation()
        # second loop
        # you read 
        # you act
        for n in range(self.nloops):
            read_t, _ = daqmx.ReadAnalogF64(self.AItask,self.SPC,self.TO,self.MODE,self.ASS,self.NCH)
            read_t = read_t/self.sensor_coff

            # moving average filter
            for ich in range(self.NICH):
                state_t[self.num_mave-1:,ich] = np.convolve(read_t[:,self.IAIs[ich]], b, mode='vaild')
            
            state_ave = ( np.average(np.split(state_t,int(self.num/tilen),0),axis=1) + self.dynamic_pressre )/self.dynamic_pressre
            seq[self.num-int(self.num/tilen):self.num,:] = np.round(state_ave,2)
            # action
            ai = self.actor.get_action(seq.reshape(-1,self.num,self.NICH), self.QN,train)
            daqmx.WriteAnalogF64(self.AOtask,self.OB,1,self.TO,daqmx.Val_GroupByChannel,self.PA[ai,:]*3)
            # reward
            reward_t  = np.average(read_t[:,self.CH100])
            self.memory.add_local(seq,ai,reward_t,0)
            read = np.append(read,read_t,axis=0)
            seq = shift(seq,[-self.num/tilen,0],cval=0)

        # third loop
        # you read 
        # you do not act
        # make sure PA turn off
        for n in range(self.nloops):
            read_t, _ = daqmx.ReadAnalogF64(self.AItask,self.SPC,self.TO,self.MODE,self.ASS,self.NCH)
            read_t = read_t/self.sensor_coff
            
            # moving average filter
            for ich in range(self.NICH):
                state_t[self.num_mave-1:,ich] = np.convolve(read_t[:,self.IAIs[ich]], b, mode='vaild')

            state_ave = ( np.average(np.split(state_t,int(self.num/tilen),0),axis=1) + self.dynamic_pressre )/self.dynamic_pressre
            seq[self.num-int(self.num/tilen):self.num,:] = np.round(state_ave,2)
            # action
            # action zero
            daqmx.WriteAnalogF64(self.AOtask,self.OB,1,self.TO,daqmx.Val_GroupByChannel,np.zeros(self.OB))
            # reward
            reward_t  = np.average(read_t[:,self.CH100])
            self.memory.add_local(seq,0,reward_t,0)
            read = np.append(read,read_t,axis=0)
            seq = shift(seq,[-self.num/tilen,0],cval=0)

        # stop DAQmx
        self.stop_DAQmx()
        # edit experience in buffer
        self.memory.edit_experience_local()
        self.save(read,Nepisode)
        total_reward = self.memory.totalreward()
        # move current experience to global buffer
        self.memory.add_global(total_reward)

        if (self.memory.len() > self.batch_num) and train:
            self.loss=self.QN.replay(self.memory,self.batch_num,self.gamma,targetQN)
            self.actor.reduce_epsilon()

            
        return total_reward, self.loss 