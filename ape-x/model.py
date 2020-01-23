
# coding: utf-8

# In[ ]:


import tensorflow as tf
from collections import deque
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input, Lambda, concatenate
from keras import backend as K


# In[ ]:


class Network():
    def __init__(self,args):

        self.frame_width = args.frame_width
        self.frame_height = args.frame_height
        self.state_length = args.state_length
        self.num_actions = args.n_actions ###

    def build_network(self):
        l_input = Input(shape=(self.state_length, self.frame_height, self.frame_width))
        conv2d = Conv2D(32,5,strides=(4,4),activation='relu', padding='same', data_format="channels_first")(l_input)
        conv2d = Conv2D(64,4,strides=(2,2),activation='relu', padding='same', data_format="channels_first")(conv2d)
        conv2d = Conv2D(64,3,strides=(1,1),activation='relu', padding='same', data_format="channels_first")(conv2d)
        fltn = Flatten()(conv2d)
        v = Dense(512, activation='relu', name="dense_v1")(fltn)
        v = Dense(1, name="dense_v2")(v)
        adv = Dense(512, activation='relu', name="dense_adv1")(fltn)
        adv = Dense(self.num_actions, name="dense_adv2")(adv)
        y = concatenate([v,adv])
        l_output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(self.num_actions,))(y)
        model = Model(inputs=l_input,outputs=l_output)

        s = tf.placeholder(tf.float32, [None, self.state_length, self.frame_height, self.frame_width])
        q_values = model(s)
        
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
    parser.add_argument('--frame_height', type=int, default=40, help='height of input frames')
    parser.add_argument('--state_length', type=int, default=10, help='number of input frames')
    parser.add_argument('--n_actions', type=int, default=5, help='number of actions')
    parser.add_argument('--n_step', type=int, default=3, help='n step bootstrap target')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')

    args                   = parser.parse_args()
    model                  = Network(args)
    s, q_values, q_network = model.build_network()
    q_network.summary()
