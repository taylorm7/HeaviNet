import tensorflow as tf
import numpy as np
import scipy.io as sio
import sys

from util import read_data, read_seed, load_matlab, write_song, read_index
from models import Model
from filter import butter_lowpass_filter
from audio import mu_trasform, analog_to_digital, digital_to_analog, mu_inverse

bits = 8
N = 2**bits
mu = float(N-1)
xmax = 1.0
xmin = -1.0
Q=(xmax-xmin)/N

def load(data_location, receptive_field, training_data):
    print("Loading", data_location, "for receptive field:", receptive_field)
    load_matlab(data_location, receptive_field, training_data)

def train(data_location, level, receptive_field, epochs, n_levels):
    print("Trainging on", data_location, "levels:", n_levels, 
          "with receptive_field:" , receptive_field)
    x_data, ytrue_data, x_list = read_data(data_location, receptive_field, level, training_data=True)
    index_list, frequency_list, song = read_index(data_location, receptive_field)
   
    fx = 44100

    net = Model( level, receptive_field, data_location , n_levels)
    
    song_length = len(ytrue_data)
    index_length = len(index_list[0])

    song_list = np.empty([n_levels, song_length, index_length], dtype=int)
    filtered_song = np.empty([song_length])
    print("Song:", song_length, "Index", index_length, "Song List", song_list.shape)
    for i in range(n_levels):
        filtered_song = butter_lowpass_filter(song, frequency_list[i]/2.0, fx)
        filtered_song = mu_trasform(filtered_song, mu, Q)
        filtered_song = analog_to_digital(filtered_song, Q)        
        #print("Filtered song", filtered_song.shape)
        #print(filtered_song)
        
        indicies = np.arange(song_length)
        indicies = np.repeat(indicies, index_length)
        indicies = np.reshape(indicies, (-1,index_length))
        indicies = indicies + index_list[i]
        #print("indicies", indicies.shape)
        #print(indicies)

        song_list[i] = filtered_song[indicies]

    print("Song List", song_list.shape)
    net.train( song_list, ytrue_data, epochs=epochs)
    net.save(close=True)

def generate(data_location, seed_location, level, receptive_field, n_levels):
    print("Generating:", level, data_location, receptive_field, n_levels)
    print("Seed:", seed_location)
    #seed_data = read_seed(seed_file)
    seed_data, seed, seed_list = read_data(seed_location, receptive_field, level, training_data=False)
    index_list, frequency_list, song = read_index(data_location, receptive_field)

    print(seed.shape)
    gen_net = Model( level, receptive_field, data_location, n_levels )
    
    sample_length = 304128
    song_data = gen_net.generate(seed, seed_list, index_list, sample_length)
    gen_net.close()
    song_name = write_song( song_data, seed_location, level, receptive_field)
    
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
