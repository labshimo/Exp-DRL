#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf
from collections import deque
import time
import pandas as pd
from simulator import Simulator
from model import Network
from property import Property


class Actor:
    def __init__(self,
                 args,
                 queues,
                 number,
                 sess,
                 param_copy_interval=100,
                 send_size=10,                  # Send () transition to shared queue in a time.
                 no_op_steps=5,                # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
                 epsilon=0.1,
                 alpha=7,
                 anealing=True,
                 no_anealing_steps=0,
                 anealing_steps=4000,
                 initial_epsilon=1.0,
                 final_epsilon=0.1):

        self.queue               = queues[0]
        self.param_queue         = queues[1]
        self.data_queue          = queues[2]
        self.buffer_queue        = queues[3]
        self.path                = args.path
        if args.train:
            self.data_path           = args.path + '/data'
            self.run_path            = args.path + '/run'
        else:
            self.data_path           = args.path + '/test'
            self.run_path            = args.path + '/test'
        self.num_episodes        = args.num_episodes
        self.num_actors          = args.num_actors
        self.frame_width         = args.frame_width
        self.frame_height        = args.frame_height
        self.state_length        = args.state_length
        self.n_step              = args.n_step
        self.gamma               = args.gamma
        self.gamma_n             = self.gamma**self.n_step
        self.param_copy_interval = param_copy_interval
        self.send_size           = send_size
        self.no_op_steps         = no_op_steps
        self.epsilon             = epsilon
        self.alpha               = alpha
        self.anealing            = anealing
        self.no_anealing_steps   = no_anealing_steps
        self.anealing_steps      = anealing_steps
        self.prop                = Property()
        self.env                 = Simulator(args, self.prop)
        self.num                 = number
        self.num_actions         = args.n_actions
        self.a_buffer            = [0] * self.num_actions
        self.t                   = 0
        self.total_reward        = 0
        self.total_q_max         = 0

        if not self.anealing:
            self.epsilon         = self.epsilon **(1+(self.num/(self.num_actors-1))*self.alpha) if self.num_actors != 1 else self.epsilon
        else:
            self.epsilon         = initial_epsilon
            self.epsilon_step    = (initial_epsilon - final_epsilon)/ anealing_steps

        self.model               = Network(args)
        self.local_memory        = deque(maxlen=self.send_size*2)
        self.env_memory          = np.zeros((0,self.env.num_i))
        self.buffer_memory       = np.zeros((0,4+self.num_actions))
       
        with tf.device("/cpu:0"):
            self.s, self.q_values, q_network = self.model.build_network()
        q_network.summary()

        self.q_network_weights   = self.bubble_sort_parameters(q_network.trainable_weights)
        with tf.device("/cpu:0"):
            self.st, self.target_q_values, target_network = self.model.build_network()
        self.target_network_weights = self.bubble_sort_parameters(target_network.trainable_weights)

        self.a, self.y, self.q, self.error = self.td_error_op()

        learner_params = self.param_queue.get()
        shapes = self.get_params_shape(learner_params)

        self.ph_list = [tf.placeholder(tf.float32, shape=shapes[i]) for i in range(len(shapes))]
        self.target_ph_list = [tf.placeholder(tf.float32, shape=shapes[i]) for i in range(len(shapes))]
        self.obtain_q_parameters = [self.q_network_weights[i].assign(self.ph_list[i]) for i in range(len(self.q_network_weights))]
        self.obtain_target_parameters = [self.target_network_weights[i].assign(self.target_ph_list[i]) for i in range(len(self.target_network_weights))]

        self.sess = sess

        self.sess.run(tf.global_variables_initializer())

        self.sess.run([self.obtain_q_parameters,self.obtain_target_parameters],
                      feed_dict=self.create_feed_dict(learner_params))
        
        self.init_model_action()

    def init_model_action(self):
        # first call of keras needs 0.22s
        state = np.zeros((self.state_length*self.frame_height,self.frame_width))
        # keras needs to drive in 0.001s
        for _ in range(5):
            s=time.time()
            q = self.q_values.eval(feed_dict={self.s: [np.float32(state)]}, session=self.sess)
            f=time.time()
            print("elapsed time = {0:05f}".format((f-s)))

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
  
    def td_error_op(self):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])
        q = tf.placeholder(tf.float32, [None,None])
        #w = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector. shape=(BATCH_SIZE, num_actions)
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        # shape = (BATCH_SIZE,)
        q_value = tf.reduce_sum(tf.multiply(q, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)

        return a, y, q, error

    def get_initial_state(self, observation):
        state = [processed_observation for _ in range(self.state_length)]
        return np.stack(state, axis=0)
    
    def get_action_and_q(self, state):
        q = self.q_values.eval(feed_dict={self.s: [np.float32(state)]}, session=self.sess)
        if self.epsilon >= random.random():
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(q[0])

        return action, q[0], np.max(q)

    def get_action_at_test(self, state):
        action = self.repeated_action

        if random.random() <= 0.05:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state)]}))

        return action

    def get_sample(self, n):
        s, a, _,  _, q  = self.buffer[0]
        _, _, _, s_, q_ = self.buffer[n-1]
        return s, a, self.R, s_, q, q_
        
    def calculate_R(self, reward):
        self.R = round((self.R + reward * self.gamma_n) / self.gamma,3)

    def calculate_n_step_transition(self):
        if len(self.buffer) >= self.n_step:
            s, a, r, s_, q, q_ = self.get_sample(self.n_step)
            self.local_memory.append((s, a, r, s_, q, q_))
            self.R = self.R - self.buffer[0][2]
            self.buffer.pop(0)
            
    def add_experience_and_priority_to_remote_memory(self):
        # Add experience and priority to remote memory
        if len(self.local_memory) > self.send_size:
            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            #terminal_batch = []
            q_batch = []
            qn_batch = []

            for _ in range(self.send_size):
                data = self.local_memory.popleft()
                state_batch.append(data[0])
                action_batch.append(data[1])
                reward_batch.append(data[2])
                #shape = (BATCH_SIZE, 4, 32, 32)
                next_state_batch.append(data[3])
                #terminal_batch.append(data[4])
                q_batch.append(data[4])
                qn_batch.append(data[5])

            # shape = (BATCH_SIZE, num_actions)
            target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch))}, session=self.sess)
            # DDQN
            actions = np.argmax(qn_batch, axis=1)
            target_q_values_batch = np.array([target_q_values_batch[i][action] for i, action in enumerate(actions)])
            # shape = (BATCH_SIZE,)
            y_batch = reward_batch + self.gamma_n * target_q_values_batch

            error_batch = self.error.eval(feed_dict={
                self.s: np.float32(np.array(state_batch)),
                self.a: action_batch,
                self.q: q_batch,
                self.y: y_batch
            }, session=self.sess)

            send = [(state_batch[i],action_batch[i],reward_batch[i],next_state_batch[i]) for i in range(self.send_size)]

            self.queue.put((send,error_batch))
            
    def copy_weight(self):
        if self.t % self.param_copy_interval == 0:
            while self.param_queue.empty():
                print('Actor {} is wainting for learner params coming'.format(self.num))
                time.sleep(4)
            learner_params = self.param_queue.get()
            self.sess.run([self.obtain_q_parameters,self.obtain_target_parameters],
                          feed_dict=self.create_feed_dict(learner_params))
        if self.anealing and self.anealing_steps + self.no_anealing_steps > (self.t/100) >= self.no_anealing_steps:
            self.epsilon -= self.epsilon_step
            
    def start_loop(self,num_loop):
        self.env.setup_DAQmx()
        # initiate
        state  = self.env.get_initial_state()
        # start read analog 
        self.env.start_reading()
        self.env.write_daqmx_zero()
        self.env.write_task.start()
        #self.env.write_daqmx_zero()
        #self.env.write_daqmx_zero()
    
        # first loop 
        for n in range(num_loop):    
                       
            # adopt output timing and action zero
            self.env.write_daqmx_zero()

            observation           = self.env.get_observation()
            processed_observation = self.env.preprocess(observation)
            next_state            = np.append(state[self.frame_height:, :], processed_observation, axis=0)
            self.env_memory       = np.append(self.env_memory,observation,axis=0)
            cpte                  = np.average(observation[:,self.env.loc_100])
            m                     = [0, cpte, 0, 0] + self.a_buffer
            # m.extend(self.a_buffer)
            self.buffer_memory    = np.append(self.buffer_memory,[m],axis=0)
            state = next_state
        return state
       
    def end_loop(self,num_loop,state):        
        # third loop
        for _ in range(num_loop):
            # adopt output timing and action zero
            self.env.write_daqmx_zero()
            self.env.write_daqmx_zero()
            observation           = self.env.get_observation()
            processed_observation = self.env.preprocess(observation)
            next_state            = np.append(state[self.frame_height:, :], processed_observation, axis=0)
            cpte                  = np.average(observation[:,self.env.loc_100])
            self.env_memory       = np.append(self.env_memory,observation,axis=0)
            m                     = [0, cpte, 0, 0] + self.a_buffer
            # m.extend(self.a_buffer)
            self.buffer_memory    = np.append(self.buffer_memory,[m],axis=0)
            state                 = next_state
        # stop DAQmx
        self.env.stop_DAQmx()
        return state
        
    def run(self):
        # for csvfile
        out = 'EPISOD,ACTOR,EPSILON,TOTAL_REWARD,MAX_Q_AVG\n'
        with open(self.path + '/actor.csv','a') as f:
            f.write(out)

        episode = 0
        self.run_one_episode(episode)
        while episode<self.num_episodes:
            try:
                self.run_one_episode(episode)
                episode += 1
            except KeyboardInterrupt:
                print("Program stopped by Ctrl+C")
                break
            except:
                print('error!')
                self.env.stop_DAQmx()
                
        print("Actor", self.num, "is Over.")
        time.sleep(0.5)

    def run_one_episode(self,episode):
        # initialize
        self.buffer       = []
        self.R            = 0
        self.total_reward = 0
        self.total_q_max  = 0
        self.env.attachment_count = 0

        # measure time
        start = time.time()
        # simulation start
        state = self.start_loop(int(self.env.n_loop/2))
        # measure time
        # ineract_with_envornment
        for n in range(self.env.n_loop):
            # action
            m_1 = time.time()
            action, q, q_max      = self.get_action_and_q(state)
            self.env.write_daqmx(action)
            m_2 = time.time()
            #print("\r{0}".format(m_2-m_1), end="")

            observation           = self.env.get_observation() 
            processed_observation = self.env.preprocess(observation)
            next_state            = np.append(state[self.frame_height:, :], processed_observation, axis=0)   
            cpte                  = np.average(observation[:,self.env.loc_100])
            reward                = self.env.get_reward_with_keep_attachment(cpte,action)
            #reward                = self.env.get_reward(cpte)
            self.buffer.append((state, action, reward, next_state, q))
            self.calculate_R(reward)
            state = next_state
            # n-step transition
            self.calculate_n_step_transition()          
            self.env_memory       = np.append(self.env_memory,observation,axis=0)
            m                     = [action, cpte, reward, np.argmax(q)]
            m.extend(q)
            self.buffer_memory    = np.append(self.buffer_memory,[m],axis=0)
            self.total_reward    += reward 
            self.total_q_max     += q_max
            self.t               += 1
        # simulation end

        state = self.end_loop(int(self.env.n_loop/2),state)
        # Add_experience_and_priority_to_remote_memory
        self.add_experience_and_priority_to_remote_memory()
        # copy weight
        self.copy_weight()
        # measure time
        elapsed = time.time() - start         
        # write text 
        text = 'EPISODE: {0:6d} / ACTOR: {1:3d} / EPSILON: {2:.5f} / TOTAL_REWARD: {3:3.0f} / MAX_Q_AVG: {4:2.4f} '.format(episode + 1, self.num, self.epsilon, self.total_reward, (self.total_q_max / float(self.env.n_loop)))          
        out="{0:6d},{1:3d},{2:.5f},{3:3.0f},{4:2.4f}\n".format(episode + 1, self.num, self.epsilon, self.total_reward, (self.total_q_max / float(self.env.n_loop)))
        print(text)
        with open(self.path + '/actor.csv','a') as f:
            f.write(out)
        # send data to copy
        self.save_env_data(episode)
        self.send_env_data()

    def send_env_data(self):
        #self.data_queue.put(self.env_memory)
        #self.buffer_queue.put(self.buffer_memory)
        self.env_memory          = np.zeros((0,self.env.num_i))
        self.buffer_memory       = np.zeros((0,4+self.num_actions))

    def save_env_data(self,episode):
        data_save_path = self.data_path + '/data{:0=5}.csv'.format(episode+1)
        run_save_path  = self.run_path + '/run{:0=5}.csv'.format(episode+1)
        df_d = pd.DataFrame(self.env_memory)
        df_r = pd.DataFrame(self.buffer_memory)
        df_d.to_csv(data_save_path)
        df_r.to_csv(run_save_path)
