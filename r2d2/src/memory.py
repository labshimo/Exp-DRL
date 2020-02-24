import gym

import pickle
import os
import numpy as np
import random
import time
import traceback
import math

import tensorflow as tf

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import *
from keras import backend as K

import multiprocessing as mp

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from memory_profiler import profile
from memory_profiler import memory_usage
#--------------------------------------------------------

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity= capacity
        self.index = 0
        self.buffer = []

    def add(self, experience, priority):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def update(self, idx, experience, priority):
        pass

    def sample(self, batch_size, steps):
        batchs = random.sample(self.buffer, batch_size)

        indexes = np.empty(batch_size, dtype='float32')
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)


#--------------------------------------------------------

import heapq
class _head_wrapper():
    def __init__(self, data):
        self.d = data
    def __eq__(self, other):
        return True

class PERGreedyMemory():
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, experience, priority):
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()
        
        # priority は最初は最大を選択
        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-priority, experience))

    def update(self, idx, experience, priority):
        # heapqは最小値を出すためマイナス
        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-priority, experience))
    
    def sample(self, batch_size, step):
        # 取り出す(学習後に再度追加)
        batchs = [heapq.heappop(self.buffer)[1].d for _ in range(batch_size)]

        indexes = np.empty(batch_size, dtype='float32')
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)

class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.0
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.max_p = 1

    def _getPriority(self, error):
        return (error + self.e) ** self.a
    
    def add(self, sample, error):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def add_p(self, p, sample):
        self.tree.add(p, sample)
    
    def sample(self, n):
        batch = []
        idx_batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idx_batch.append(idx)

        return idx_batch, batch

    def update(self, idx, error):
        p = self._getPriority(error)
        if p > self.max_p:
            self.max_p = p
        self.tree.update(idx, p)

    def update_batch(self, idx_batch, error_batch):
        p_batch = self._getPriority(error_batch)
        if np.max(p_batch) > self.max_p:
            self.max_p = np.max(p_batch)
        self.tree.update_batch(idx_batch, p_batch)
    
    def __len__(self):
        return self.tree.length

    def __del__(self):
        self.tree.initialize()
        print("reset")


#copy from https://github.com/jaromiru/AI-blog/blob/5aa9f0b/SumTree.py
import numpy

class SumTree:
    write = 0
    length = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.initialize()

    def initialize(self):
        self.tree = numpy.zeros( 2*self.capacity - 1 )
        self.data = numpy.zeros( self.capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write  += 1
        
        if self.write >= self.capacity:
            self.write = 0
        
        self.length += 1
        self.length  = min(self.length, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PERProportionalMemory():
    def __init__(self, capacity, beta_initial, beta_steps, enable_is):
        self.capacity = capacity
        self.tree = SumTree(capacity)

        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is

    def add(self, experience, priority):
        self.tree.add(priority, experience)

    def update(self, index, experience, priority):
        self.tree.update(index, priority)

    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
    
        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        total = self.tree.total()
        section = total / batch_size
        for i in range(batch_size):
            r = section*i + random.random()*section
            (idx, priority, experience) = self.tree.get(r)

            indexes.append(idx)
            batchs.append(experience)

            if self.enable_is:
                # 重要度サンプリングを計算
                weights[i] = (self.capacity * priority / total) ** (-beta)
            else:
                weights[i] = 1  # 無効なら1

        if self.enable_is:
            # 安定性の理由から最大値で正規化
            weights = weights / weights.max()

        return (indexes ,batchs, weights)

    def __len__(self):
        return self.tree.length

#------------------------------------

import bisect
class _bisect_wrapper():
    def __init__(self, data):
        self.d = data
        self.priority = 0
        self.p = 0
    def __lt__(self, o):  # a<b
        return self.priority > o.priority

class PERRankBaseMemory():
    def __init__(self, capacity, alpha, beta_initial, beta_steps, enable_is):
        self.capacity = capacity
        self.buffer = []
        self.alpha = alpha
        
        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is

    def add(self, experience, priority):
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()
        
        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)

    def update(self, index, experience, priority):
        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)


    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする。
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps

        total = 0
        for i, o in enumerate(self.buffer):
            o.index = i
            o.p = (len(self.buffer) - i) ** self.alpha 
            total += o.p
            o.p_total = total

        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        index_lst = []
        section = total / batch_size
        rand = []
        for i in range(batch_size):
            rand.append(section*i + random.random()*section)
        
        rand_i = 0
        for i in range(len(self.buffer)):
            if rand[rand_i] < self.buffer[i].p_total:
                index_lst.append(i)
                rand_i += 1
                if rand_i >= len(rand):
                    break

        for i, index in enumerate(reversed(index_lst)):
            o = self.buffer.pop(index)  # 後ろから取得するのでindexに変化なし
            batchs.append(o.d)
            indexes.append(index)

            if self.enable_is:
                # 重要度サンプリングを計算
                priority = o.p
                weights[i] = (self.capacity * priority / total) ** (-beta)
            else:
                weights[i] = 1  # 無効なら1

        if self.enable_is:
            # 安定性の理由から最大値で正規化
            weights = weights / weights.max()

        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)

#----------------------------------------