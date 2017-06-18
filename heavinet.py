import tensorflow as tf
import numpy as np
import scipy.io as sio

from util import read_data, name_level, format_level
from models import Model

receptive_field = 5
n_levels = 7

x_names, ytrue_names = name_level(n_levels)

print x_names, ytrue_names

x_data, ytrue_data = read_data(receptive_field, x_names, ytrue_names, force_read=False)

for xn, ytn in zip(x_names, ytrue_names):
    print x_data[xn].size, ytrue_data[ytn].size


net = Model( 0, receptive_field, x_data[x_names[0]], ytrue_data[ytrue_names[0]] ) 
net.train(x_data[x_names[0]], ytrue_data[ ytrue_names[0] ] )


'''
nets = []
for i in range(n_levels):
    nets.append( Model( 
                    i,
                    receptive_field,
                    x_names[i], 
                    ytrue_names[i]) )

for net in nets:
    net.train()
'''
