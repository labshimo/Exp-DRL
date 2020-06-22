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

#---------------------------------------------------
# network
#---------------------------------------------------
def clipped_error_loss(y_true, y_pred):
    err = y_true - y_pred  # エラー
    L2 = 0.5 * K.square(err)
    L1 = K.abs(err) - 0.5

    # エラーが[-1,1]区間ならL2、それ以外ならL1を選択する。
    loss = tf.where((K.abs(err) < 1.0), L2, L1)   # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

def rescaling(x, epsilon=0.001):
    n = math.sqrt(abs(x)+1) - 1
    return np.sign(x)*n + epsilon*x

def rescaling_inverse(x):
    return np.sign(x)*( (x+np.sign(x) ) ** 2 - 1)

def build_compile_model(
    batch_size, 
    input_sequence,
    input_shape,             # 入力shape
    enable_image_layer,      # image_layerを入れるか
    nb_actions,              # アクション数
    enable_dueling_network,  # dueling_network を有効にするか
    dueling_network_type,
    enable_noisynet,
    dense_units_num,         # Dense層のユニット数
    lstm_units_num,          # LSTMのユニット数
    metrics,                 # compile に渡す metrics
    ):

    c = input_ = Input(batch_shape=(batch_size,input_sequence, input_shape[0], input_shape[1]))

    if enable_image_layer:
        # (time steps, w, h) -> (time steps, w, h, ch)
        c = Reshape((input_sequence, input_shape[0], input_shape[1], 1))(c)
        # https://keras.io/layers/wrappers/
        c = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="same"), name="c1")(c)
        c = Activation("relu")(c)
        c = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="same"), name="c2")(c)
        c = Activation("relu")(c)
        c = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding="same"), name="c3")(c)
        c = Activation("relu")(c)
        c = TimeDistributed(Flatten())(c)
        c = LSTM(lstm_units_num, name="lstm", stateful=True)(c)

    if enable_dueling_network:
        # value
        v = Dense(dense_units_num, activation="relu")(c)
        if enable_noisynet:
            v = NoisyDense(1, name="v")(v)
        else:
            v = Dense(1, name="v")(v)

        # advance
        adv = Dense(dense_units_num, activation='relu')(c)
        if enable_noisynet:
            adv = NoisyDense(nb_actions, name="adv")(adv)
        else:
            adv = Dense(nb_actions, name="adv")(adv)

        # 連結で結合
        c = Concatenate()([v,adv])
        if dueling_network_type == "ave":
            c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_actions,))(c)
        elif dueling_network_type == "max":
            c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_actions,))(c)
        elif dueling_network_type == "naive":
            c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_actions,))(c)
        else:
            raise ValueError('dueling_network_type is ["ave","max","naive"]')
    else:
        c = Dense(dense_units_num, activation="relu")(c)
        if enable_noisynet:
            c = NoisyDense(nb_actions, activation="linear", name="adv")(c)
        else:
            c = Dense(nb_actions, activation="linear", name="adv")(c)
    
    
    model = Model(input_, c)

    # compile
    model.compile(
        loss=clipped_error_loss, 
        optimizer=Adam(lr=0.00025), 
        metrics=metrics)
    
    return model
