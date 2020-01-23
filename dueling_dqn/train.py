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
from keras.models import model_from_json
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras import backend as K
import tensorflow as tf 
import csv
from datetime import datetime
from scipy.ndimage.interpolation import shift 
import argparse
from actor import Actor
from memory import Memory
from simulator import Simulator
from logger import Logger
from model import Dueling_CNN

if __name__=="__main__":

    '''
    argparse start 
    '''

    default_modeltype = "CNN"
    default_simtype   = "single"
    current_id        = '../data/'+datetime.now().strftime('%B%d%H%M%S')
    parser            = argparse.ArgumentParser(description='Experient')
    # set output directry name 
    parser.add_argument("-dir", type=str, default="log",help="output dir for loggin, default:'log'")
    # set number of epoisodes
    parser.add_argument("-ne", type=int, default=500, help="number of episodes, default:2000")
    # set number of steps in 1-episode
    parser.add_argument("-ns", type=int, default=300, help="number of steps, default:300")
    # set gamma value
    parser.add_argument("-gamma", type=float, default=0.99, help="gamma, default:0.99")
    # set epsilon value
    parser.add_argument("-epsilon", type=float, default=1.0, help="epsilon for survey, default:1.0")
    # set differential epsilon value
    parser.add_argument("-deps", type=float, default=0.0005, help="differential epsilon for survey, default:0.005")
    # set input size
    parser.add_argument("-s", type=int, default=40, help="number of agent input default:100")
    # set hidden layer's size
    parser.add_argument("-hi", type=int, default=64, help="hidden layer size default:128")
    # set action size 
    parser.add_argument("-a", type=int, default=3, help="output layer size default:7")
    # set learning rate
    parser.add_argument("-lr", type=float, default=0.0001, help="learning rate default:0.01")
    # set memory size
    parser.add_argument("-memory", type=int, default=100*100, help="memory size for experience replay default:100*200")
    # set batch size
    parser.add_argument("-batch", type=int, default=32,help="replay batch size, default:500")
    # set update frequncy of target model
    parser.add_argument("-upfre", type=int, default=3, help="update frequncy of target model default:3")
    # set model type
    parser.add_argument("-model", type=str, default=default_modeltype,
            help="ModelType, default:'"+default_modeltype+"'")
    # set simulator type
    parser.add_argument("-sim", type=str, default=default_simtype,
            help="ModelType, default:'"+default_simtype+"'")
    # check
    parser.add_argument("-y", type=int, default=0,
            help="OK?, default:0")
    args = parser.parse_args()

    print(args)
    if args.y == 0:
        input("OK?")
    '''
    argparse end 
    '''

    ## Make directory and write setting log
    if not os.path.exists(current_id):
        os.makedirs(current_id)
        os.makedirs(current_id+'/runs')
        os.makedirs(current_id+'/runs/state')
        os.makedirs(current_id+'/runs/state_')
        os.makedirs(current_id+'/runs/action')
        os.makedirs(current_id+'/runs/reward')
        os.makedirs(current_id+'/runs/cpte')
        os.makedirs(current_id+'/runs/punish')
        os.makedirs(current_id+'/origin')
        os.makedirs(current_id+'/trained_model')
    with open(os.path.join(current_id+'/log'+".args"), "w") as argsf:
        argsf.write(str(args))

    np.random.seed(0)

    logger   = Logger('../data/summary/'+datetime.now().strftime('%B%d%H%M%S'))
    # parameters ### do not touch ###
    # analog input 
    AIfre               = 20000.0 # freaquency
    samps_per_chan      = 200  # buffer
    array_size_samps    = 200 # buffer 
    phys_chan_I1        = "Dev1/ai0"
    phys_chan_I2        = "Dev1/ai1"
    phys_chan_I3        = "Dev1/ai2"
    phys_chan_I4        = "Dev1/ai3"
    phys_chan_I5        = "Dev1/ai4"
    phy_channels        = [phys_chan_I1,phys_chan_I2,phys_chan_I3,phys_chan_I4,phys_chan_I5]
    input_channels      = [0,3,4]
    chnane_num_100      = 4
    sensor_coff         = np.array([22.984,23.261,22.850,35.801,25.222])/1000000*900
    fillmode            = daqmx.Val_GroupByChannel
    # analog output
    AOfre               = 12000.0 # frequency
    AObu                = 120 # buffer
    phy_chan_O          = "Dev1/ao0"
    # another parameters
    timeout             = 10
    dt                  = 1/AIfre
    totaltime           = 1
    max_number_of_steps = int(AIfre*totaltime/samps_per_chan)
    ## Init model
    
    mainQN   = Dueling_CNN(current_id,args.lr,args.s,len(input_channels),args.a,args.hi)   
    targetQN = Dueling_CNN(current_id,args.lr,args.s,len(input_channels),args.a,args.hi)   
    memory   = Memory(args.s,args.memory,args.ns,len(input_channels))

    actor = Actor(args.epsilon,args.deps,args.a)
    mainQN.model.summary()
    
    # import nondimentional frequency
    PA = np.zeros((actor.n_actions,AObu))
    with open('csv/PA120-5.csv', 'r') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            PA[i,:] = row

   
    sim   = Simulator(args.s,actor,mainQN,memory,args.gamma,args.batch,AIfre,samps_per_chan,array_size_samps,
                        fillmode,phy_channels,input_channels,chnane_num_100,sensor_coff,phy_chan_O,AOfre,
                        AObu,timeout,max_number_of_steps,PA,current_id)




    # learning loop start
    #mainQN.load_model('model1000.json','weights1000.hdf5')
    sim.init_model_action()
    highscore = 0
    fw=open(current_id+"/log.csv","w")
    for i in range(args.ne):
        if i%args.upfre==0:
            targetQN = mainQN 
        
        if i%10!=0:
            total_reward,loss=sim.run(i, targetQN,train=True)
        else:
            total_reward,loss=sim.run(i, targetQN,train=False)

            if highscore<total_reward:
                print('Hiscore!')
                highscore=total_reward
                modelname = 'model/model%d.ckpt'%i
                mainQN.save_model(i)
            aw=memory.total_reward_award
            out=("%d,%d,%f,%2.2e,%d,%d\n" % (i,total_reward,loss,actor.epsilon,np.min(aw),np.max(aw)))
            logger.log_scalar('/Total Reward/Episode',total_reward,i)
            logger.log_scalar('/Average Loss/Episode',loss,i)
            for x in range(len(mainQN.model.layers)):
                try:
                    w=mainQN.model.layers[x].get_weights()[0]
                    b=mainQN.model.layers[x].get_weights()[1]
                    # give to tensorboard
                    histname = 'layer%d'%x
                    logger.log_histogram(histname + 'weight', w,i)
                    logger.log_histogram(histname + 'bias',   b,i)
                except:
                    break
            #for log
            print("EPOCH:%d/%d|REWARD:%d|LOSS:%f|EPSILON:%0.2f|MIN:%d|MAX:%d"
                    %(i,args.ne,total_reward,loss,actor.epsilon,np.min(aw),np.max(aw)))
            fw.write(out)
            fw.flush()
        '''
        if i%500 ==0:
            input('input anything to continue')
        '''
    mainQN.save_model(args.ne*2)
    # test
    for test_i in range(3):
        test_reward,_=sim.run(args.ne+test_i, targetQN,train=False)
        print("result%d" %(test_reward))
    #end training
    #save model
    
