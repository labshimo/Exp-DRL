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

import rl.core

import multiprocessing as mp

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# from : https://github.com/LuEE-C/Noisy-A3C-Keras/blob/master/NoisyDense.py
# from : https://github.com/keiohta/tf2rl/blob/atari/tf2rl/networks/noisy_dense.py
class NoisyDense(Layer):

    def __init__(self, units,
                 sigma_init=0.02,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.kernel_shape = tf.constant((self.input_dim, self.units))
        self.bias_shape = tf.constant((self.units,))

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.sigma_kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=initializers.Constant(value=self.sigma_init),
                                      name='sigma_kernel'
                                      )


        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.sigma_bias = self.add_weight(shape=(self.units,),
                                        initializer=initializers.Constant(value=self.sigma_init),
                                        name='sigma_bias')
        else:
            self.bias = None
            self.epsilon_bias = None

        self.epsilon_kernel = K.zeros(shape=(self.input_dim, self.units))
        self.epsilon_bias = K.zeros(shape=(self.units,))

        self.sample_noise()
        super(NoisyDense, self).build(input_shape)


    def call(self, X):
        #perturbation = self.sigma_kernel * self.epsilon_kernel
        #perturbed_kernel = self.kernel + perturbation
        perturbed_kernel = self.sigma_kernel * K.random_uniform(shape=self.kernel_shape)

        output = K.dot(X, perturbed_kernel)
        if self.use_bias:
            #bias_perturbation = self.sigma_bias * self.epsilon_bias
            #perturbed_bias = self.bias + bias_perturbation
            perturbed_bias = self.bias + self.sigma_bias * K.random_uniform(shape=self.bias_shape)
            output = K.bias_add(output, perturbed_bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def sample_noise(self):
        K.set_value(self.epsilon_kernel, np.random.normal(0, 1, (self.input_dim, self.units)))
        K.set_value(self.epsilon_bias, np.random.normal(0, 1, (self.units,)))

    def remove_noise(self):
        K.set_value(self.epsilon_kernel, np.zeros(shape=(self.input_dim, self.units)))
        K.set_value(self.epsilon_bias, np.zeros(shape=self.units,))


#---------------------------------------------------
