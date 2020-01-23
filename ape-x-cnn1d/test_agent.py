#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf
from collections import deque
import time

from simulator import Simulator
from model import Network
from property import Property

# In[ ]:


class Agent:
    def __init__(self,
                 args,
                 queues,
                 sess):

        self.queue               = queues[0]
        self.param_queue         = queues[1]
        self.data_queue          = queues[2]
        self.buffer_queue        = queues[3]
        self.load                = args.load
        self.save_network_path   = args.path + '/saved_networks/'
        self.path                = args.path
        self.num_episodes        = args.test_num
        self.frame_width         = args.frame_width
        self.frame_height        = args.frame_height
        self.state_length        = args.state_length
        self.prop                = Property()
        self.env                 = Simulator(args, self.prop)
        self.t                   = 0
        self.total_reward        = 0
        self.total_q_max         = 0
        self.model               = Network(args)
        self.env_memory          = np.zeros((0,self.env.num_i))
        self.buffer              = []
        self.num_actions         = args.n_actions
        self.buffer_memory       = np.zeros((0,3+self.num_actions))
        self.a_buffer            = [0] * self.num_actions

        with tf.variable_scope("learner_parameters", reuse=False):
            with tf.device("/cpu:0"):
                self.s, self.q_values, q_network = self.model.build_network()

        self.q_network_weights   = self.bubble_sort_parameters(q_network.trainable_weights)

        self.sess = sess

        self.sess.run(tf.global_variables_initializer())

        with tf.device("/cpu:0"):
            self.saver = tf.train.Saver(self.q_network_weights)

        # Load network
        if self.load:
            self.load_network()

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.save_network_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')
        
    def create_feed_dict(self, learner_params):
        feed_dict = {}
        for i in range(len(learner_params[0])):
            feed_dict[self.ph_list[i]] = learner_params[0][i]
            feed_dict[self.target_ph_list[i]] = learner_params[1][i]
        return feed_dict

    def get_params_shape(self, learner_params):
        shapes = []
        for p in learner_params[0]:
            shapes.append(p.shape)
        return shapes
    
    def bubble_sort_parameters(self, arr):
        change = True
        while change:
            change = False
            for i in range(len(arr) - 1):
                if arr[i].name > arr[i + 1].name:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    change = True
        return arr

    def get_initial_state(self, observation):
        state = [processed_observation for _ in range(self.state_length)]
        return np.stack(state, axis=0)
    
    def get_action_at_test(self, state):
        q = self.q_values.eval(feed_dict={self.s: [np.float32(state)]}, session=self.sess)
        action = np.argmax(q[0])
        
        return action, q[0], np.max(q)
             
    def start_loop(self,num_loop):
        self.env.setup_DAQmx()
        # initiate
        state  = self.env.get_initial_state()
        # start read analog 
        self.env.start_reading()
        self.env.write_daqmx_zero()
        self.env.write_task.start()
        self.env.write_daqmx_zero()
        self.env.write_daqmx_zero()

        # first loop 
        for n in range(num_loop):    
            # adopt output timing and action zero
            self.env.write_daqmx_zero()

            observation = self.env.get_observation()
            processed_observation = self.env.preprocess(observation)
            next_state            = np.append(state[self.frame_height:, :], processed_observation, axis=0)
            # adopt output timing and action zero
            self.env_memory    = np.append(self.env_memory,observation,axis=0)
            cpte               = np.average(observation[:,self.env.loc_100])
            m                  = [0, cpte, 0]
            m.extend(self.a_buffer)
            self.buffer_memory = np.append(self.buffer_memory,[m],axis=0)
            state = next_state
        return state
    
    def end_loop(self,num_loop,state):        
        # third loop
        for _ in range(num_loop):
            # adopt output timing and action zero
            self.env.write_daqmx_zero()
            self.env.write_daqmx_zero()
            observation = self.env.get_observation()
            processed_observation = self.env.preprocess(observation)
            next_state            = np.append(state[self.frame_height:, :], processed_observation, axis=0)           
            self.env_memory    = np.append(self.env_memory,observation,axis=0)
            cpte               = np.average(observation[:,self.env.loc_100])
            m                  = [0, cpte, 0]
            m.extend(self.a_buffer)
            self.buffer_memory = np.append(self.buffer_memory,[m],axis=0)
            state = next_state
        # stop DAQmx
        self.env.stop_DAQmx()
        
    def run(self):
        for episode in range(self.num_episodes):
            # initialize
            self.total_reward = 0
            self.total_q_max  = 0
            # simulation start
            state = self.start_loop(int(self.env.n_loop/2))
            # measure time
            start = time.time()
            # ineract_with_envornment
            for n in range(self.env.n_loop):
                # action
                action,q,q_max        = self.get_action_at_test(state)
                self.env.write_daqmx(action)

                observation           = self.env.get_observation()
                processed_observation = self.env.preprocess(observation)
                next_state            = np.append(state[self.frame_height:, :], processed_observation, axis=0)       
                cpte                  = np.average(observation[:,self.env.loc_100])
                #reward                = self.env.get_reward_with_punish(cpte,action)
                reward                = self.env.get_reward(cpte)
                self.buffer.append((state, action, reward, next_state, q))
                state = next_state
                
                self.env_memory    = np.append(self.env_memory,observation,axis=0)
                m = [action, cpte, reward]
                m.extend(q)
                self.buffer_memory = np.append(self.buffer_memory,[m],axis=0)

                self.total_reward += reward 
                self.total_q_max  += q_max
                self.t            += 1
            # simulation end
            state = self.end_loop(int(self.env.n_loop/2),state)
            # measure time
            elapsed = time.time() - start         
            # write text 
            text = 'EPISODE: {0:6d} / TOTAL_REWARD: {1:3.0f}'.format(episode + 1, self.total_reward)
            print(text)   

            with open(self.path + '/test.txt','a') as f:
                f.write(text+"\n")
            
            self.send_env_data()


    def send_env_data(self):
        self.data_queue.put(self.env_memory)
        self.buffer_queue.put(self.buffer_memory)
        self.buffer_memory       = np.zeros((0,3+self.num_actions))
        self.env_memory = np.zeros((0,self.env.num_i))
