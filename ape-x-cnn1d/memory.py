#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import time
from memory_profiler import profile
from sumtree import SumTree

class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.max_p = 1
        self.e = 0.0
        self.a = 0.6

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def length(self):
        return self.tree.write

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

        return batch, idx_batch

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

