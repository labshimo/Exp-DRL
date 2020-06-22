import gym
import pickle
import os
import numpy as np
import random
import time
import traceback
import math
import json
import multiprocessing as mp

import matplotlib.pyplot as plt
from collections import deque
from network import *

#---------------------------------------------------
# actor
#---------------------------------------------------
class Actor():
    def __init__(self, 
        simlator,
        actor_index,
        epsilon,
        model_args,
        args,
        experience_q,
        model_sync_q,
        training = True,
        **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.actor_index               = actor_index
        if not args["test"]:
            self.epsilon               = epsilon
        else:
            self.epsilon               = 0
        self.experience_q              = experience_q
        self.model_sync_q              = model_sync_q
        self.nb_actions                = simlator.nb_actions
        self.input_shape               = args["input_shape"]
        self.input_sequence            = args["input_sequence"]
        self.actor_model_sync_interval = args["actor_model_sync_interval"]
        self.gamma                     = args["gamma"] 
        self.n_step                    = args["multireward_steps"]
        self.action_interval           = args["action_interval"]
        self.enable_dueling_network    = args["enable_dueling_network"]
        self.enable_noisynet           = args["enable_noisynet"]
        self.enable_rescaling_priority = args["enable_rescaling_priority"]
        self.rescaling_epsilon         = args["rescaling_epsilon"]
        self.priority_exponent         = args["priority_exponent"]
        self.per_alpha                 = args["per_alpha"]
        if not args["test"]:
            self.episode_num          = args["training_number"]
        else:
            self.episode_num          = args["test_number"]
        self.burn_in_length            = args["burnin_length"]
        self.sequence_length           = args["sequence_length"]
        self.overlap_length            = args["overlap_length"]
        self.no_op_steps               = args["no_op_steps"]
        self.gamma_n                   = self.gamma**self.n_step
        # local memory
        self.buffer                    = []
        self.local_buffer              = []
        self.hidden_state_buffer       = []
        self.R                         = 0
        # model
        self.model    = build_compile_model(**model_args)
        self.lstm     = self.model.get_layer("lstm")
        self.compiled = True
        # training
        self.training = training
        # simlator
        self.simlator = simlator
                
        # for log
        self.log               = args["log_dir"] + args["actor_log"]
        self.total_reward      = 0
        self.total_loss        = 0
        self.duration          = 0
        self.episode           = 0
        self.step              = 0
        self.total_q_max       = []
        self.elapsed           = 0
        self.csv_index         = 0
        self.repeated_action   = 0
        self.repeated_q_values = np.zeros(self.nb_actions)
        self.repeated_q_max    = 0
        self.init_model_action()

    def init_model_action(self):
        # first call of keras needs 0.22s
        zero_state = np.zeros((self.input_sequence,self.input_shape[0],self.input_shape[1]))
        # keras needs to drive in 0.001s
        for _ in range(5):
            s=time.time()
            q = self._get_qmax(zero_state)
            f=time.time()
            print("elapsed time = {0:05f}".format((f-s)))

    def run_actor(self):        
        while self.episode < self.episode_num:
            try:
                self.run_simlator()
            except:
                print('error!')
                self.simlator.stop()
                
        print("Actor", self.actor_index, "is Over.")
        time.sleep(0.1)

    def get_initial_state(self, observation):
        state     = [np.zeros(self.input_shape) for _ in range(self.input_sequence)]
        state[-1] = observation
        return state

    def get_next_state(self, state, observation):
        next_state = state[1:]
        next_state.append(observation)
        return next_state

    def reset_simlator(self):
        terminal = False
        self.model.reset_states()
        observation = self.simlator.reset()
        state       = self.get_initial_state(observation)
        # batch(state, action, reward, next_state, priority)
        self.sequantial_data  = [deque([], self.burn_in_length+self.sequence_length) for _ in range(7)] 
        self.timing_send_data = 0
        return state, terminal

    def run_simlator(self):
        start = time.time()
        state, terminal = self.reset_simlator()
        for _ in range(random.randint(20, self.no_op_steps)):
            observation, _, _ = self.simlator.step(self.nb_actions, np.zeros(self.nb_actions))  # Do nothing
            state = self.get_next_state(state,observation) 
        time_arr = []
        while not terminal:
            action, q_values, q_max, hidden_state = self.forward(state)
            observation, reward, terminal = self.simlator.step(action, q_values)
            next_state = self.get_next_state(state,observation)
            self.append_local_memory(state, action, reward, next_state, terminal, hidden_state, q_values)
            self.backward(reward, terminal)
                 
            state = next_state
            self.total_q_max.append(q_max) # for log
            self.total_reward += reward
            self.duration     += 1
            self.step         += 1
            
        for _ in range(self.no_op_steps):
            observation, _, _ = self.simlator.step(self.nb_actions, np.zeros(self.nb_actions))  # Do nothing
            state = self.get_next_state(state,observation) 
        self.simlator.stop()
        self.elapsed = time.time() - start
        self.print_log()

    def compile(self, optimizer, metrics=[]):
        self.compiled = True

    def load_weights(self, filepath):
        print("WARNING: Not Loaded. Please use 'load_weights_path' param.")

    def save_weights(self, filepath, overwrite=False):
        print("WARNING: Not Saved. Please use 'save_weights_path' param.")

    def forward(self, state):
        # set hidden state for sequantial batch
        hidden_state = [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]

        # フレームスキップ(action_interval毎に行動を選択する)
        action, q_values, q_max = self.repeated_action, self.repeated_q_values, self.repeated_q_max 

        if self.step % self.action_interval == 0:
            # 行動を決定
            action, q_values, q_max = self.select_action(state)
            # リピート用
            self.repeated_action, self.repeated_q_values, self.repeated_q_max = action, q_values, q_max

        return action, q_values, q_max, hidden_state

    # 長いので関数に
    def select_action(self,state):
        # noisy netが有効の場合はそちらで探索する
        q_values = self._get_qmax(state)
        if self.training and not self.enable_noisynet:
            # ϵ-greedy法
            if self.epsilon > np.random.uniform(0, 1):
                # ランダム
                action = np.random.randint(0, self.nb_actions)
            else:
                action = np.argmax(q_values)
        else:
            action = np.argmax(q_values)
        return action, q_values, np.max(q_values)

    # 2箇所あるので関数に、現状の最大Q値のアクションを返す
    def _get_qmax(self,state):
        q_values = self.model.predict(np.asarray([state]), batch_size=1)[0]
        
        return q_values

    def backward(self, reward, terminal):
        if self.training: 
            while len(self.local_buffer) > 0:
                [batch.append(data) for batch, data in zip(self.sequantial_data, self.local_buffer.pop())]
                self.timing_send_data += 1

            if self.timing_send_data == (self.burn_in_length+self.sequence_length) or terminal:
                burn_in_data, batch_data = [], [[] for _ in range(7)] 
                [burn_in_data.append(self.sequantial_data[0][i]) for i in range(self.burn_in_length)]
                [batch_data[i].append(self.sequantial_data[i][self.burn_in_length+j]) for i in range(7) for j in range(self.sequence_length)]
                priority = self.assemble_priority(batch_data[6])

                send_data = ((
                        batch_data[0],
                        batch_data[1],
                        batch_data[2], 
                        batch_data[3],
                        batch_data[4][0],
                        batch_data[5][0],
                        burn_in_data,
                        priority,
                    ))
                self.experience_q.put(send_data)
                self.timing_send_data -= (self.burn_in_length+self.overlap_length)
                
            # 一定間隔で model を learner からsyncさせる
            if self.step % self.actor_model_sync_interval == 0:
                # 要求を送る
                self.model_sync_q[0].put(1)  # 要求
            
            # weightが届いていれば更新
            if not self.model_sync_q[1].empty():
                weights = self.model_sync_q[1].get(timeout=1)
                # 空にする(念のため)
                while not self.model_sync_q[1].empty():
                    self.model_sync_q[1].get(timeout=1)
                self.model.set_weights(weights)

        return []

    def append_local_memory(self, state, action, reward, next_state, terminal, hidden_state, q_values):
        if self.training:       
            self.buffer.append((state, action, reward, next_state, hidden_state[0], hidden_state[1], q_values))
            self.R = round((self.R + reward * self.gamma_n) / self.gamma,3)

            # n-step transition
            if terminal: # terminal state
                while len(self.buffer) > 0:
                    n = len(self.buffer)
                    s, a, r, s_, h0, h1, q, q_ = self.get_sample(n)
                    p = self.get_priority(a, r, q, q_ )
                    # add to local memory
                    self.local_buffer.append((s, a, r, s_, h0, h1, p))
                    self.R = round((self.R - self.buffer[0][2]) / self.gamma,3)
                    self.buffer.pop(0)
                self.R = 0

            if len(self.buffer) >= self.n_step:
                s, a, r, s_, h0, h1, q, q_  = self.get_sample(self.n_step) 
                p = self.get_priority(a, r, q, q_ )
                # add to local memory
                self.local_buffer.append((s, a, r, s_, h0, h1, p))
                self.R = self.R - self.buffer[0][2]
                self.buffer.pop(0)

    def get_sample(self, n):
        s, a, _, _, h0, h1, q  = self.buffer[0]
        _, _, _, s_ , _, _, q_ = self.buffer[n-1]
        # state, action, R, next state, q value, next q value, hidden state
        return s, a, self.R, s_, h0, h1, q, q_

    def get_priority(self, action, reward, q_value, target_q_value):
        # priority のために TD-error をだす。
        # 現在のQネットワークを出力
        target_q_max = np.max(target_q_value)
        # priority計算
        if self.enable_rescaling_priority:
            td_error = rescaling(target_q_max) ** -1
        td_error = reward + self.gamma_n * target_q_max
        if self.enable_rescaling_priority:
            td_error = rescaling(td_error, self.rescaling_epsilon)
        priority = abs(td_error - q_value[action]) ** self.per_alpha

        return priority

    def assemble_priority(self, priorities):
        return self.priority_exponent * np.max(priorities) + (1-self.priority_exponent) * np.average(priorities)  

    def rescaling(self, x, epsilon=0.001):
        n = math.sqrt(abs(x)+1) - 1
        return np.sign(x)*n + epsilon*x

    def rescaling_inverse(self, x):
        return np.sign(x)*( (x+np.sign(x) ) ** 2 - 1)
        
    def print_log(self):
        text = 'EPISODE: {0:6d} / ACTOR: {1:3d} / TOTAL_STEPS: {2:8d} / STEPS: {3:5d} / EPSILON: {4:.5f} / TOTAL_REWARD: {5:3.0f} / MAX_Q_AVG: {6:2.4f} / STEPS_PER_SECOND: {7:.1f}'.format(
            self.episode + 1, self.actor_index, self.step, self.duration, self.epsilon,
            self.total_reward, np.average(self.total_q_max),
            self.duration/self.elapsed)

        print(text)

        with open(self.log,'a') as f:
            f.write(text+"\n")

        self.total_reward = 0
        self.total_q_max  = 0
        self.total_loss   = 0
        self.duration     = 0
        self.episode     += 1
        self.total_q_max  = []
        self.csv_index    = 0

        self.simlator.save_simlator_data(self.episode)

    def write_csv(self, start, actions, rewards, priorities):
        logname = "a{0}e{1}.csv".format(self.actor_index,self.episode)

        with open(logname,'a') as f:
            f.write("sequential:{0}\n".format(self.csv_index))
            f.write("index, action, reward, priority \n")             
            for i, (action, reward, priority) in enumerate(zip(actions, rewards, priorities)):
                text = "{0}, {1}, {2}, {3}".format(i+start, action, reward, priority)
                f.write(text+"\n")
        self.csv_index += 1

    @property
    def layers(self):
        return self.model.layers[:]

def test(epsilon):
    # build_compile_model 関数用の引数
    with open("DRL/R2D2/src/args.json","r") as f:
        args = json.load(f) 
    print(args["input_sequence"])
    print(args["input_shape"])

    model_actor_args = {
        "batch_size": 1,
        "input_sequence": 4,
        "input_shape": (84,84),
        "enable_image_layer": args["enable_image_layer"],
        "nb_actions": args["nb_actions"],
        "enable_dueling_network": args["enable_dueling_network"],
        "dueling_network_type": args["dueling_network_type"],
        "enable_noisynet": args["enable_noisynet"],
        "dense_units_num": args["dense_units_num"],
        "lstm_units_num": args["lstm_units_num"],
        "metrics": args["metrics"],
        "create_optimizer_func": create_optimizer(),
    }
    experience_q = mp.Queue()
    model_sync_q = [mp.Queue(), mp.Queue(), mp.Queue()]
    with tf.device("/device:CPU:0"):
        actor = Actor(
            0,
            epsilon,
            model_actor_args,
            args,
            experience_q,
            model_sync_q,
        )
        
        # start
        actor.run_actor()

def create_optimizer():
    return Adam(lr=0.00025)


if __name__ == '__main__':
    test(0.5)