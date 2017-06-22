import tensorflow as tf
import numpy as np
import scipy.io as sio
import sys

from util import read_data, name_level, read_seed, load_matlab, read_2, write_song
from models import Model

def load(data_location, receptive_field):
    print "Loading", data_location, "for receptive field:", receptive_field
    load_matlab(data_location, receptive_field)


def train(data_location, level, receptive_field, epochs):
    print "Trainging on ", data_location, "with receptive_field:" , receptive_field
    
    x_data, ytrue_data = read_2(data_location, receptive_field, level)
    
    net = Model( level, receptive_field, data_location )
    net.train( x_data, ytrue_data, epochs=epochs )
    net.save(close=True)


def generate(data_location, seed_file, level, receptive_field, seed):
    print "Generating:", level, data_location, seed_file, receptive_field, seed
    if seed == True :
        print "Seed:", seed_file
        seed_data = read_seed(seed_file, receptive_field)
        gen_net = Model( level, receptive_field, data_location )
        song_data = gen_net.generate(seed_data)
        song_name = write_song( song_data, data_location, level, receptive_field)
        return song_name
    else:
        print "not seed"

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
            if sys.argv[6] == 'True':
                generate(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), seed=True)
            else:
                generate(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), seed=False)

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
