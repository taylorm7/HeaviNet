import tensorflow as tf
import numpy as np
import scipy.io as sio
import sys

from util import read_data, read_seed, load_matlab, write_song, read_index
from models import Model
from filter import butter_lowpass_filter
from audio import format_song, quantize, test_songs

def load(data_location, receptive_field, training_data):
    print("Loading", data_location, "for receptive field:", receptive_field)
    load_matlab(data_location, receptive_field, training_data)

def train(data_location, level, receptive_field, epochs, n_levels):
    index_list, frequency_list, song, fx = read_index(data_location, receptive_field)
    print("Trainging on", data_location, "levels:", n_levels, 
          "with receptive_field:" , receptive_field, "fx", fx)
    #test_songs(song, frequency_list, n_levels, data_location)
    
    net = Model( level, receptive_field, data_location , n_levels)
    
    bits = 8
    song_length = len(song)
    song_list = format_song(song, frequency_list, index_list, song_length, n_levels, data_location, bits, fx)
    ytrue = quantize(song, bits=bits)
    
    '''
    t_start = 2500
    t_end = 2510
    print("Song", song[t_start:t_end].flatten() )
    print("Ytrue",ytrue.shape, ytrue[t_start:t_end].flatten() )
    print("Song List", song_list.shape )
    for i in range(n_levels):
        print("Level",i,song_list[i,t_start:t_end,:] )
    '''
    net.train( song_list, ytrue, epochs=epochs)
    net.save(close=True)

def generate(data_location, seed_location, level, receptive_field, n_levels):
    print("Generating:", level, data_location, receptive_field, n_levels)
    print("Seed:", seed_location)
    #seed_data = read_seed(seed_file)
    #seed_data, seed, seed_list = read_data(seed_location, receptive_field, level, training_data=False)
    index_list, frequency_list, song, fx = read_index(seed_location, receptive_field)

    gen_net = Model( level, receptive_field, data_location, n_levels )
    
    sample_length = 36000
    #sample_length = 1000
    song_data, epochs = gen_net.generate(song, index_list, frequency_list, sample_length)

    gen_net.close()
    song_name = write_song( song_data[-sample_length:-1] , fx, seed_location, level, receptive_field, epochs)

def predict(data_location, seed_location, level, receptive_field, n_levels):
    print("Predicting:", level, data_location, receptive_field, n_levels)
    print("Seed:", seed_location)
    index_list, frequency_list, song, fx = read_index(seed_location, receptive_field)

    gen_net = Model( level, receptive_field, data_location, n_levels )
    
    sample_length = 2000
    song_data, epochs = gen_net.generate(song, index_list, frequency_list, sample_length)

    gen_net.close()
    song_name = write_song( song_data[-sample_length:-1] , fx, seed_location, level, receptive_field, epochs)


    
if __name__ == '__main__':
    if len(sys.argv) >= 3:
        if sys.argv[1]=='load': 
            load(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) )
        elif sys.argv[1]=='train':
            train(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
        elif sys.argv[1]=='generate':
            generate(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
        elif sys.argv[1]=='predict':
            predict(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
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
