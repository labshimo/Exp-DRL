import gym

import pickle
import os
import numpy as np
import random
import time
import traceback
import math
import json
import tensorflow as tf

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import *
from keras import backend as K

import rl.core

import multiprocessing as mp

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from noisedense import NoisyDense
from memory import ReplayMemory, PERGreedyMemory, SumTree, PERProportionalMemory, PERRankBaseMemory
from agent import R2D2Manager
import sys

#-----------------------------------------------------
# NN可視化用
#-----------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import cv2
#-----------------------------------------------------------
# main    
#-----------------------------------------------------------

def load_args(filename):
    with open(filename,"r") as f:
        args = json.load(f) 
    return args

def save_args(args):
    fw = open(args["agent"]["log_dir"]+'args-log.json','w')
    json.dump(args,fw,indent=4)

def main(image):
    global agent
       
    args = load_args(sys.argv[1])
    save_args(args)
    
    manager = R2D2Manager(num_actors=args["agent"]["num_actors"], args=args)
    manager.train()

if __name__ == '__main__':
    main(image=True)