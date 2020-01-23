#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import time
import pandas as pd

class Saver:
    def __init__(self,
                 args,
                 queues,
                 max_queue_no_added=30
                 ):

        self.queue               = queues[0]
        self.param_queue         = queues[1]
        self.data_queue          = queues[2]
        self.buffer_queue        = queues[3]
        self.max_queue_no_added  = max_queue_no_added
        self.no_added_count      = 0
        self.path                = args.path
        if not args.train:
        	self.data_path           = args.path + '/data'
        	self.run_path            = args.path + '/run'
        else:
        	self.data_path           = args.path + '/test'
        	self.run_path            = args.path + '/teet'
        self.num_episodes        = args.num_episodes
        self.i_episodes          = 0
        self.env_memory_list     = deque()
        self.run_memory_list     = deque()

    def save_env_data(self):
        data_save_path = self.data_path + '/data{:0=5}.csv'.format(self.i_episodes)
        run_save_path  = self.run_path + '/run{:0=5}.csv'.format(self.i_episodes)
        data = self.env_memory_list.popleft()
        run  = self.run_memory_list.popleft()
        df_d = pd.DataFrame(data)
        df_r = pd.DataFrame(run)
        df_d.to_csv(data_save_path)
        df_r.to_csv(run_save_path)
        self.i_episodes += 1
        print('episode{:0=5} is saved'.format(self.i_episodes))
    def run(self):
        print("Saver Starts! ")
        while self.no_added_count < self.max_queue_no_added:
            if self.i_episodes == 0:
            	if self.data_queue.empty():
            		time.sleep(5)
            		print("Saver is waiting!")
            	else:
            		data_queue   = self.data_queue.get()
            		buffer_queue = self.buffer_queue.get()
            		self.env_memory_list.append(data_queue)
            		self.run_memory_list.append(buffer_queue)
            		self.save_env_data()
            		time.sleep(1)
            else:
                if self.data_queue.empty():
                    self.no_added_count += 1
                    time.sleep(1)
                else:
                    self.no_added_count  = 0
                    data_queue   = self.data_queue.get()
                    buffer_queue = self.buffer_queue.get()
                    self.env_memory_list.append(data_queue)
                    self.run_memory_list.append(buffer_queue)
                    self.save_env_data()	
                    time.sleep(1)
        print("Saver end")
        	        


      