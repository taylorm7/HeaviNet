import tensorflow as tf
import numpy as np
import scipy.io as sio
import sys

from util import read_data, read_seed, load_matlab, write_song
from models import Model

def load(data_location, receptive_field):
    print "Loading", data_location, "for receptive field:", receptive_field
    load_matlab(data_location, receptive_field)


def train(data_location, level, receptive_field, epochs):
    print "Trainging on ", data_location, "with receptive_field:" , receptive_field
    
    x_data, ytrue_data = read_data(data_location, receptive_field, level)
    
    net = Model( level, receptive_field, data_location )
    net.test(x_data, ytrue_data)
    net.close()
    #net.train( x_data, ytrue_data, epochs=epochs )
    #net.save(close=True)


def generate(data_location, seed_file, level, receptive_field):
    print "Generating:", level, data_location, seed_file, receptive_field
    print "Seed:", seed_file
    seed_data = read_seed(seed_file, receptive_field)
    gen_net = Model( level, receptive_field, data_location )
    song_data = gen_net.generate(seed_data)
    gen_net.close()
    song_name = write_song( song_data, data_location, level, receptive_field)

    #net = Model( level, receptive_field, data_location )
    #net.close()
    #nets = []
    #for i in range(7):
        #nets.append( Model( i,receptive_field, data_location) )
 

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        if sys.argv[1]=='load': 
            load(sys.argv[2], receptive_field=int(sys.argv[3]) )
        elif sys.argv[1]=='train':
            train(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) )
        elif sys.argv[1]=='generate':
            generate(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
        else:
            print "Invalid call, use:"
            print " heavinet load [/path/to/data] [receptive field]"
            print " heavinet train [/path/to/data] [receptive field]"
            print " heavinet generate [/path/to/data] [/path/to/seed/] [level] [receptive_field]"
    else:
        print "Invalid call, not enough arguments. Use:"
        print " heavinet load [/path/to/data] [receptive field]"
        print " heavinet train [/path/to/data] [receptive field]"
        print " heavinet generate [/path/to/data] [/path/to/seed/] [level] [receptive_field]"
