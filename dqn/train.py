# coding: utf-8
from __future__ import division, print_function
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
from model import CNN, CNN1D, Dueling_CNN
import psutil

if __name__=="__main__":

    '''
    argparse start 
    '''

    default_modeltype = "CNN"
    default_simtype   = "single"
    root_directory    = "I:/experiment/shimomura/dqn/data/cnn1d_dqn/"
    id_date           = datetime.now().strftime('%B%d%H%M%S')
    current_id        = root_directory+id_date
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
    parser.add_argument("-deps", type=float, default=0.0005, help="differential epsilon for survey, default:0.0005")
    # set input size
    parser.add_argument("-s", type=int, default=40, help="number of agent input default:100")
    # set hidden layer's size
    parser.add_argument("-hi", type=int, default=64, help="hidden layer size default:128")
    # set action size 
    parser.add_argument("-a", type=int, default=3, help="output layer size default:7")
    # set learning rate
    parser.add_argument("-lr", type=float, default=0.0001, help="learning rate default:0.0001")
    # set memory size
    parser.add_argument("-memory", type=int, default=100*100, help="memory size for experience replay default:100*100")
    # set batch size
    parser.add_argument("-batch", type=int, default=32,help="replay batch size, default:32")
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
        os.makedirs(current_id+'/runs/result')
        os.makedirs(current_id+'/origin')
        os.makedirs(current_id+'/trained_model')
    with open(os.path.join(current_id+'/log'+".args"), "w") as argsf:
        argsf.write(str(args))

    '''
    num_episodes = 2000              # 総試行回数
    max_number_of_steps = 300        # 1試行のstep数
    gamma = 0.99                     # 割引係数
    hidden_size = 128                 # Q-networkの隠れ層のニューロンの数
    learning_rate = 0.01             # Q-networkの学習係数
    memory_size = 100*200            # バッファーメモリの大きさ
    batch_size = 500                 # Q-networkを更新するバッチのSIZE   
    action_size = 7                  # 過去何コマを見るか
    update_frequency = 3
    test_highscore=0
    ''' 
    

    
    logger   = Logger(root_directory +"summary/"+id_date)
    # parameters ### do not touch ###
    # analog input 
    sample_rate         = 12000.0 # freaquency
    number_of_samples   = 120     # buffer
    input_channels      = "Dev1/ai0:4"
    state_channels      = [0,3,4]
    loc_100             = 4
    sensor_coff         = np.array([22.984,23.261,22.850,35.801,25.222])/1000000*900 #11160 11172 11125 11120 11211
    #sensor_coff         = np.array([35.801,25.222])/1000000*900 # 11120 11211
    # analog output
    output_channel      = "Dev1/ao0"
    # another parameters
    dt                  = 1/sample_rate
    totaltime           = 1
    max_number_of_steps = int(sample_rate*totaltime/number_of_samples)

    ## Init model
    
    mainQN   = CNN1D(current_id,args.lr,args.s,len(state_channels),args.a,args.hi)   
    targetQN = CNN1D(current_id,args.lr,args.s,len(state_channels),args.a,args.hi)   
    memory   = Memory(args.s,args.memory,args.ns,len(state_channels))

    actor = Actor(args.epsilon,args.deps,args.a)
    mainQN.model.summary()
    
    # import nondimentional frequency
    PA = np.zeros((actor.n_actions,number_of_samples))
    with open('csv/PA120-2.csv', 'r') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            PA[i,:] = row

   
    sim   = Simulator(args.s,args.a,actor,mainQN,memory,args.gamma,args.batch,sample_rate,number_of_samples,
        input_channels,state_channels,loc_100,sensor_coff,output_channel,max_number_of_steps,PA,current_id)



    # learning loop start
    mainQN.load_model('model1000.json','weights1000.hdf5')
    sim.init_model_action()
    highscore = 0
    fw=open(current_id+"/log.csv","w")
    #LEARNING START 
    learning_start = time.time()  

    for i in range(args.ne):
        if i%args.upfre==0:
            targetQN = mainQN 
        
        if i%10!=0:
            total_reward,loss,q_max=sim.run(i, targetQN,train=False)
        else:
            total_reward,loss,q_max=sim.run(i, targetQN,train=False)

            if highscore<total_reward:
                print('Hiscore!')
                highscore=total_reward
                modelname = 'model/model%d.ckpt'%i
                mainQN.save_model(i)
            aw=memory.total_reward_award
            
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
            print("EPOCH:%d/%d|REWARD:%d|LOSS:%f|QMAX:%f|EPSILON:%0.2f|MIN:%d|MAX:%d"
                    %(i,args.ne,total_reward,loss,q_max,actor.epsilon,np.min(aw),np.max(aw)))
        out=("%d,%d,%f,%f,%2.2e,%d,%d\n" % (i,total_reward,loss,q_max,actor.epsilon,np.min(aw),np.max(aw)))
        fw.write(out)
        fw.flush()
        '''
        if i%500 ==0:
            input('input anything to continue')
        '''
    mainQN.save_model(args.ne*2)

    #LEARNING END 
    learning_end = time.time()  
    print("learning takes %d minutes" %(int((learning_end-learning_start)/60)))

    # test
    for test_i in range(3):
        total_reward,loss,q_max=sim.run(args.ne+test_i, targetQN,train=False)
        print("result%d" %(total_reward))
    #end training
    #save model
    
