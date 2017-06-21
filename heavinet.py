import tensorflow as tf
import numpy as np
import scipy.io as sio
import sys

from util import read_data, name_level, read_seed, load_matlab, read_2
from models import Model

def load(data_location, receptive_field=7):
    print "Loading", data_location, "for receptive field:", receptive_field
    load_matlab(data_location, receptive_field)


def train(data_location, receptive_field=7, force_read=False):
    print "Trainging on ", data_location, "with receptive_field:" , receptive_field
    
    x_data, ytrue_data = read_2(data_location, receptive_field, 1)
    
    net = Model( 1, receptive_field, data_location )
    net.train( x_data, ytrue_data, epochs=1 )
    net.save(close=True)


'''
    n_levels = 7
    
    x_names, ytrue_names = name_level(n_levels)

    print x_names, ytrue_names

    x_data, ytrue_data = read_data(receptive_field, x_names, ytrue_names, 
                                    data_location, force_read=force_read)

    for xn, ytn in zip(x_names, ytrue_names):
        print x_data[xn].size, ytrue_data[ytn].size

    net = Model( 1, receptive_field, data_location )
    net.train( x_data[x_names[1]], ytrue_data[ytrue_names[1]], epochs=5 )
    net.save(close=True)


    #net = Model( 1, receptive_field, data_location )

    for i in range(n_levels):
        print type(i)
        net =  Model( i,receptive_field, data_location)
        net.train( x_data[ x_names[i] ], ytrue_data[ ytrue_names[i] ], epochs=1 )
        net.save(close=True)
'''     
        #nets.append( Model( i,receptive_field, data_location) )
    
    #for net, xn, ytn in zip(nets, x_names, ytrue_names):
    #    net.train( x_data[xn], ytrue_data[ytn], epochs=1 )
    #for net in nets:
    #    net.save(close=True)

def generate(data_location, seed_file, level, receptive_field=7):
    print "Generating:", level, data_location, seed_file, receptive_field
    seed_data = read_seed(seed_file)
    print seed_data.size

    print type(1) 
    net = Model( 1, receptive_field, data_location )
    net.close()
    #nets = []
    #for i in range(7):
        #nets.append( Model( i,receptive_field, data_location) )
 

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        if sys.argv[1]=='load':
            if len(sys.argv) >= 4 and sys.argv[3].isdigit():
                print "Input receptive field:", int(sys.argv[3])
                load(sys.argv[2], receptive_field= int(sys.argv[3]) )
            else:
                print "Default receptive field"
                load(sys.argv[2])
        elif sys.argv[1]=='train':
            if len(sys.argv) >= 4 and sys.argv[3].isdigit():
                print "Input receptive field:", int(sys.argv[3])
                train(sys.argv[2], receptive_field= int(sys.argv[3]) )
            else:
                print "Default receptive field"
                train(sys.argv[2])
        elif sys.argv[1]=='generate':
            if len(sys.argv) >= 6 and sys.argv[5].isdigit():
                generate(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
            else:
                generate(sys.argv[2], sys.argv[3], int(sys.argv[4]) )

        else:
            print "Invalid call, use: heavinet train /path/to/data train [receptive field]"
            print "or: heavinet generate /path/to/data /path/to/seed/"
    else:
        print "Please supply more  arguments; 'heavinet /path/to/data train [receptive field]"
        print "or 'heavinet /path/to/data generate /path/to/seed/"
'''
nets = []
for i in range(n_levels):
    nets.append( Model( 
                    i,
                    receptive_field,
                    x_data[x_names[i]], 
                    ytrue_data[ytrue_names[i]]) )

for net in nets:
    net.train(epochs=10)
'''
