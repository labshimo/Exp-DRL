#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
import time

from model import Network
from memory import Memory

# In[ ]:


class Learner:
    def __init__(self,
                 args,
                 queues,
                 sess,
                 target_update_interval=2500,
                 batch_size=512,
                 lr=0.00025/4,
                 save_interval=100,
                 print_interval=100,
                 max_queue_no_added=1000
                 ):

        self.path                    = args.path
        self.load                    = args.load
        self.save_path               = args.path + '/saved_networks/'
        self.replay_memory_size      = args.replay_memory_size
        self.initial_memory_size     = args.initial_memory_size
        self.frame_width             = args.frame_width
        self.frame_height            = args.frame_height
        self.state_length            = args.state_length
        self.gamma_n                 = args.gamma**args.n_step
        self.queue                   = queues[0]
        self.param_queue             = queues[1]
        self.target_update_interval  = target_update_interval
        self.batch_size              = batch_size
        self.lr                      = lr
        self.save_interval           = save_interval
        self.print_interval          = print_interval
        self.max_queue_no_added      = max_queue_no_added
        self.no_added_count          = 0
        self.remote_memory           = Memory(self.replay_memory_size)
        self.model                   = Network(args)
        self.num_actions             = args.n_actions
        self.t                       = 0
        self.total_time              = 0
        self.queue_not_changed_count = 0
        # Parameters used for summary
        self.total_reward            = 0
        self.total_q_max             = 0
        self.total_loss              = 0
        self.duration                = 0
        self.episode                 = 0
        self.start                   = 0

        #with tf.device('/gpu:0'):
        with tf.variable_scope("learner_parameters", reuse=True):
                self.s, self.q_values, q_network = self.model.build_network()

        self.q_network_weights = self.bubble_sort_parameters(q_network.trainable_weights)

        # Create target network
        with tf.variable_scope("learner_target_parameters", reuse=True):
            self.st, self.target_q_values, target_network = self.model.build_network()
        self.target_network_weights = self.bubble_sort_parameters(target_network.trainable_weights)

        # Define target network update operation
        self.update_target_network = [self.target_network_weights[i].assign(self.q_network_weights[i]) for i in range(len(self.target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.error_abs, self.loss, self.grad_update, self.gv, self.cl = self.build_training_op(self.q_network_weights)

        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

        #with tf.device("/cpu:0"):
        self.saver = tf.train.Saver(self.q_network_weights)

        # Load network
        if self.load:
            self.load_network()

        params = self.sess.run((self.q_network_weights, self.target_network_weights))
        while not self.param_queue.full():
            self.param_queue.put(params)

        # Initialize target network
        self.sess.run(self.update_target_network)

    def bubble_sort_parameters(self, arr):
        change = True
        while change:
            change = False
            for i in range(len(arr) - 1):
                if arr[i].name > arr[i + 1].name:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    change = True
        return arr

    def huber_loss(self, x, delta=1.0):
        return tf.where(
            tf.abs(x) < delta,
            tf.square(x) * 0.5,
            delta * (tf.abs(x) - 0.5 * delta)
        )

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])
        #w = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector. shape=(BATCH_SIZE, num_actions)
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        # shape = (BATCH_SIZE,)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        td_error = tf.stop_gradient(y) - q_value
        errors = self.huber_loss(td_error)
        loss = tf.reduce_mean(errors)

        optimizer = tf.train.RMSPropOptimizer(self.lr, decay=0.95, epsilon=1.5e-7, centered=True)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=q_network_weights)
        capped_gvs = [(grad if grad is None else tf.clip_by_norm(grad, clip_norm=40), var) for grad, var in grads_and_vars]
        grad_update = optimizer.apply_gradients(capped_gvs)

        return a, y, tf.abs(td_error), loss, grad_update ,grads_and_vars, capped_gvs

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def run(self):#, server):

        if self.remote_memory.length() < self.initial_memory_size:
            print("Learner is Waiting... Replay memory have {} transitions".format(self.remote_memory.length()))

            while not self.queue.empty():
                t_error = self.queue.get()
                for i in range(len(t_error[0])):
                    self.remote_memory.add(t_error[0][i], t_error[1][i])

            if not self.param_queue.full():
                params = self.sess.run((self.q_network_weights, self.target_network_weights))
                while not self.param_queue.full():
                    self.param_queue.put(params)

            time.sleep(4)
            return self.run()

        print("Learner Starts!")
        while self.no_added_count < self.max_queue_no_added:
            start = time.time()

            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            w_batch = []

            if self.queue.empty():
                self.no_added_count += 1
            else:
                self.no_added_count = 0
                while not self.queue.empty():
                    t_error = self.queue.get()
                    for i in range(len(t_error[0])):
                        self.remote_memory.add(t_error[0][i], t_error[1][i])

            if not self.param_queue.full():
                params = self.sess.run((self.q_network_weights, self.target_network_weights))
                while not self.param_queue.full():
                    self.param_queue.put(params)

            minibatch, idx_batch = self.remote_memory.sample(self.batch_size)

            for data in minibatch:
                state_batch.append(data[0])
                action_batch.append(data[1])
                reward_batch.append(data[2])
                #shape = (BATCH_SIZE, 4, 32, 32)
                next_state_batch.append(data[3])

                self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(data[0])]},session=self.sess))

            # shape = (BATCH_SIZE, num_actions)
            target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch))}, session=self.sess)
            # DDQN
            actions = np.argmax(self.q_values.eval(feed_dict={self.s: np.float32(np.array(next_state_batch))}, session=self.sess), axis=1)
            target_q_values_batch = np.array([target_q_values_batch[i][action] for i, action in enumerate(actions)])
            # shape = (BATCH_SIZE,)
            y_batch = reward_batch + self.gamma_n * target_q_values_batch

            error_batch = self.error_abs.eval(feed_dict={
                self.s: np.float32(np.array(state_batch)),
                self.a: action_batch,
                self.y: y_batch
            }, session=self.sess)

            loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
                self.s: np.float32(np.array(state_batch)),
                self.a: action_batch,
                self.y: y_batch
                #self.w: w_batch
            })

            self.total_loss += loss
            self.total_time += time.time() - start

            # Memory update
            [self.remote_memory.update(idx_batch[i],error_batch[i]) for i in range(len(idx_batch))]

            self.t += 1

            if self.t % self.print_interval == 0:
                text_l = 'AVERAGE LOSS: {0:.5F} / AVG_MAX_Q: {1:2.4F} / LEARN PER SECOND: {2:.1F} / NUM LEARN: {3:5d}'.format(
                    self.total_loss/self.print_interval, self.total_q_max/(self.print_interval*self.batch_size), self.print_interval/self.total_time, self.t)
                print(text_l)
                with open(self.path + '/_output.txt','a') as f:
                    f.write(text_l+"\n")
                #print("Average Loss: ", self.total_loss/PRINT_LOSS_INTERVAL, " / Learn Per Second: ", PRINT_LOSS_INTERVAL/self.total_time, " / AVG_MAX_Q", self.total_q_max/(PRINT_LOSS_INTERVAL*BATCH_SIZE))
                self.total_loss  = 0
                self.total_time  = 0
                self.total_q_max = 0

            # Update target network
            if self.t % self.target_update_interval == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % self.save_interval == 0:
                with tf.device('/cpu:0'):
                    save_path = self.saver.save(self.sess, self.save_path, global_step=(self.t))
                print('Successfully saved: ' + save_path)

        print("The Learning is Over.")
        time.sleep(0.5)

