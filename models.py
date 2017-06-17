import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, level, receptive_field, x_name, ytrue_name,
                       batch_size=128, 
                       n_nodes_hl1=500,
                       n_nodes_hl2=500,
                       n_nodes_hl3=500):
        self.level = level
        self.x_name = x_name
        self.ytrue_name = ytrue_name
        
        clip_size = 2*receptive_field+1
        n_input_classes = 2**level
        n_target_classes = 2**(level+1)

        inputs = tf.placeholder(tf.int32, [None,clip_size])


    def train(self, x, ytrue):
        print "Trainging:",  self.level, self.x_name , x.size, self.ytrue_name, ytrue.size 
        
