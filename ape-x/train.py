
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import argparse
import time
import random

from actor import Actor
from learner import Learner
from simulator import Simulator
from saver import Saver
from test_agent import Agent


def actor_work(args, queues, num):
    # with tf.device('/cpu:0'):
    sess = tf.InteractiveSession()
    actor = Actor(args, queues, num, sess, param_copy_interval=2000, send_size=200)
    actor.run()

def leaner_work(args, queues):
    # with tf.device('/gpu:0'):
    sess = tf.InteractiveSession()
    leaner = Learner(args, queues, sess, batch_size=126)
    leaner.run()

def saver_work(args, queues):
    # with tf.device('/gpu:0'):
    saver = Saver(args, queues)
    saver.run()

def agent_work(args,queues):
    # with tf.device('/cpu:0'):
    sess = tf.InteractiveSession()
    actor = Agent(args, queues, sess)
    actor.run()

    
# Train Mode
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_actors', type=int, default=1, help='number of Actors')
    parser.add_argument('--train', type=int, default=1, help='train mode or test mode')
    parser.add_argument('--load', type=int, default=0, help='loading saved network')
    parser.add_argument('--path', type=str, default=0, help='used in loading and saving (default: \'saved_networks/<env_name>\')')
    parser.add_argument('--replay_memory_size', type=int, default=2000000, help='replay memory size')
    parser.add_argument('--initial_memory_size', type=int, default=2000, help='Learner waits until replay memory stores this number of transition')
    parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes each agent plays')
    parser.add_argument('--test_num', type=int, default=5, help='number of episodes each agent plays')
    parser.add_argument('--frame_width', type=int, default=5, help='width of input frames')
    parser.add_argument('--frame_height', type=int, default=40, help='height of input frames')
    parser.add_argument('--state_length', type=int, default=10, help='number of input frames')
    parser.add_argument('--n_actions', type=int, default=5, help='number of actions')
    parser.add_argument('--n_step', type=int, default=3, help='n step bootstrap target')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')

    args = parser.parse_args()

    if args.path == 0:
        args.path = 'default'
    if not args.load:
        assert not os.path.exists(args.path), args.path+' already exists.'
    if not os.path.exists(args.path):
        os.makedirs(args.path)
        os.makedirs(args.path+'/saved_networks')
        os.makedirs(args.path+'/data')
        os.makedirs(args.path+'/run')

    if args.train:
        assert not os.path.exists(args.path + '/output.txt'), 'Output file already exists. Change file name.'

    if args.train:
        transition_queue = mp.Queue(100)
        param_dict       = mp.Queue(args.num_actors)
        data_queue       = mp.Queue()
        buffer_queue     = mp.Queue()
        ps = [mp.Process(target=leaner_work, args=(args, (transition_queue, param_dict, data_queue, buffer_queue)))]

        for i in range(args.num_actors):
            ps.append(mp.Process(target=actor_work, args=(args, (transition_queue, param_dict, data_queue, buffer_queue), i)))

        ps.append(mp.Process(target=saver_work, args=(args, (transition_queue, param_dict, data_queue, buffer_queue))))

        for p in ps:
            p.start()
            time.sleep(0.5)

        for p in ps:
            p.join()

    # Test Mode
    else:
        transition_queue = mp.Queue(100)
        param_dict       = mp.Queue()
        data_queue       = mp.Queue()
        buffer_queue     = mp.Queue()
        ps = [mp.Process(target=agent_work, args=(args, (transition_queue, param_dict, data_queue, buffer_queue)))]
        ps.append(mp.Process(target=saver_work, args=(args, (transition_queue, param_dict, data_queue, buffer_queue))))
        
        for p in ps:
            p.start()
            time.sleep(0.5)

        for p in ps:
            p.join()
    


