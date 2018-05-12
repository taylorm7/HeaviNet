import tensorflow as tf
import numpy as np
import scipy.io as sio
import sys
import os

from util import read_data, read_seed, load_matlab, write_song, read_index
from models import Model
from filter import butter_lowpass_filter
from audio import filter_song, format_song, quantize


# load: load audio file and store in pkl file for use in python
# inputs:
# data_location, string value of data directory
# receptive_field, integer value of the receptive field
# training_data, bool value (True if training data, False if seed/generation data)
# outputs:
# stores pkl file in the data location
def load(data_location, receptive_field, training_data):
    print("Loading", data_location, "for receptive field:", receptive_field)
    load_matlab(data_location, receptive_field, training_data)

# train: trains the given level according to pkl training data stored in the load function
# inputs
# data_location, string value of data directory
# level, specific level to train
# receptive_field, integer value of the receptive field
# epochs, number of epochs to train
# n_levels, total number of levels for heavinet
# outputs:
# saved neural network on trained data in data_location
def train(data_location, level, receptive_field, epochs, n_levels):
    index_list, frequency_list, song, fx = read_index(data_location, receptive_field)
    print("Trainging on", data_location, "levels:", n_levels, 
          "with receptive_field:" , receptive_field)
    net = Model( level, receptive_field, data_location , n_levels)
    
    bits = 8
    # create inputs and targets for the neural network
    level_song = filter_song(song, frequency_list, level)
    target_song = filter_song(song, frequency_list, level+1)
    # format the audio according to indicies
    song_list = format_song(level_song, frequency_list, index_list, n_levels, bits, fx)
    # quantize audio, using mu law companding
    x = quantize(level_song, bits=bits)
    ytrue = quantize(target_song, bits=bits)
    
    # train neural network
    net.train( ytrue, x, epochs=epochs)
    net.save(close=True)
# generate: generates audio given for a given level, uses either previous levels generation or seed audio
# inputs
# data_location, string value of data directory
# seed_location, string value of seed directory
# level, specific level to train
# receptive_field, integer value of the receptive field
# n_levels, total number of levels for heavinet
# outputs:
# writes out generated audio for given level to seed directory, stored in a .mat file 
def generate(data_location, seed_location, level, receptive_field, n_levels):
    print("Generating:", level, data_location, receptive_field, n_levels)
    print("Seed:", seed_location)

    index_list, frequency_list, song, fx = read_index(seed_location, receptive_field)
    gen_net = Model( level, receptive_field, data_location, n_levels )
    
     
    bits = 8
    # filter and format seed value
    seed = filter_song(song, frequency_list, level)
    seed_list = format_song(seed, frequency_list, index_list, n_levels, bits, fx)
    
    # read number of epochs trained on, used to distinguish runs at different number of epochs
    epochs = gen_net.n_epochs.eval(session=gen_net.sess)
    seed_name = os.path.split(seed_location)[1].split('.')[0]
    training_name = (os.path.split( os.path.split(seed_location)[0])[1]).split('.')[0]
    song_name = gen_net.name + "_" + str(epochs) + "_" + training_name + "_" + seed_name 
    seed_name = gen_net.seed_name + "_" + str(epochs) + "_" + training_name + "_" + seed_name 

    # either read from previous level or from seed value
    if level != 0 and os.path.isfile(seed_location + '/' + seed_name + '.mat'):
        print("Reading Sead")
        seed_data = read_seed(seed_name, seed_location)
    else:
        print("Using seed")
        seed_data = seed
    print("Seed", seed_data.shape)

    # quantize data, feed into neural network, and write to .mat file
    seed_data = quantize(seed_data, bits=bits)
    song_data, epochs = gen_net.generate(seed_data)
    gen_net.close()
    song_name = write_song( song_data, fx, seed_location, song_name)
    
if __name__ == '__main__':
    if len(sys.argv) >= 3:
        if sys.argv[1]=='load': 
            load(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) )
        elif sys.argv[1]=='train':
            train(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
        elif sys.argv[1]=='generate':
            generate(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
        else:
            print("Invalid call, use:")
            print(" heavinet load [/path/to/data] [receptive field]")
            print(" heavinet train [/path/to/data] [receptive field]")
            print(" heavinet generate [/path/to/data] [/path/to/seed/] [level] [receptive_field]")
    else:
        print("Invalid call, not enough arguments. Use:")
        print(" heavinet load [/path/to/data] [receptive field]")
        print(" heavinet train [/path/to/data] [receptive field]")
        print(" heavinet generate [/path/to/data] [/path/to/seed/] [level] [receptive_field]")
