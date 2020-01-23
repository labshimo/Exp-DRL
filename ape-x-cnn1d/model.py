
# coding: utf-8

# In[ ]:


import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Lambda, concatenate, BatchNormalization, Activation
from keras import backend as K
from memory_profiler import profile

class Network():
    def __init__(self,args,hidden_size=32):

        self.frame_width  = args.frame_width
        self.frame_height = args.frame_height
        self.state_length = args.state_length
        self.num_actions  = args.n_actions 
        self.hidden_size  = hidden_size
    def build_network(self):
        # 1st convolutional layer block
        l_input  = Input(shape=(self.state_length*self.frame_height,self.frame_width))
        conv1d = Conv1D(self.hidden_size,10,padding='same', name='cnn1d1-1')(l_input)
        conv1d = MaxPooling1D(pool_size=4)(conv1d)
        conv1d = Activation('relu')(conv1d)
        conv1d = Conv1D(self.hidden_size,10,padding='same', name='cnn1d1-2')(conv1d)
        conv1d = BatchNormalization(name='Batch1')(conv1d)
        conv1d = Activation('relu')(conv1d)
        conv1d = MaxPooling1D(pool_size=4)(conv1d)
        # full conected layer
        fltn = Flatten()(conv1d)
        v = Dense(self.hidden_size, activation='relu')(fltn)
        v = Dense(1)(v)
        adv = Dense(self.hidden_size, activation='relu')(fltn)
        adv = Dense(self.num_actions)(adv)
        y = concatenate([v,adv])
        l_output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), 
            output_shape=(self.num_actions,))(y)

        model       = Model(inputs=l_input,outputs=l_output)
        s           = tf.placeholder(tf.float32, [None, self.state_length*self.frame_height, self.frame_width])
        q_values    = model(s)
        
        return s, q_values, model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_actors', type=int, default=1, help='number of Actors')
    parser.add_argument('--train', type=int, default=1, help='train mode or test mode')
    parser.add_argument('--load', type=int, default=0, help='loading saved network')
    parser.add_argument('--path', type=str, default=0, help='used in loading and saving (default: \'saved_networks/<env_name>\')')
    parser.add_argument('--replay_memory_size', type=int, default=2000000, help='replay memory size')
    parser.add_argument('--initial_memory_size', type=int, default=200, help='Learner waits until replay memory stores this number of transition')
    parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes each agent plays')
    parser.add_argument('--frame_width', type=int, default=5, help='width of input frames')
    parser.add_argument('--frame_height', type=int, default=10, help='height of input frames')
    parser.add_argument('--state_length', type=int, default=10, help='number of input frames')
    parser.add_argument('--n_actions', type=int, default=5, help='number of actions')
    parser.add_argument('--n_step', type=int, default=3, help='n step bootstrap target')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')

    args                   = parser.parse_args()
    model                  = Network(args)
    s, q_values, q_network = model.build_network()
    q_network.summary()
