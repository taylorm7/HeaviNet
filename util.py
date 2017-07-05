import numpy as np
import os.path
import scipy.io
import h5py
import hdf5storage

import cPickle as pkl

def name_level(n_levels):
    x_name = []
    ytrue_name = []
    for i in range(0,n_levels):
        x_name.append(str( "level_" + str(i)))
        ytrue_name.append(str("ytrue_" + str(i)))
    return x_name, ytrue_name


def format_level(iterable, receptive_field):
    indicies = range(receptive_field-1, -1, -1)
    for i in range(len(indicies)):
        indicies[i] = -1* 2**indicies[i]
    indicies.append(0)
    indicies_pos = range(0, receptive_field)
    for i in range(len(indicies_pos)):
        indicies_pos[i] = 2**indicies_pos[i]
    
    indicies = indicies + indicies_pos
    
    level = []
    for i in range(len(iterable)):
        l = []
        for index in indicies:
            l.extend( iterable[ (i+index)%len(iterable) ])
        level = np.append(level, l)
    
    return level

def load_matlab(data_location, receptive_field):
    read_file = data_location + "/matlab_song_r" + str(receptive_field) + ".mat"
    
    try: 
        os.path.isfile(read_file)
    except IOError:
        print "Failed to load:", store_file

    with h5py.File(read_file) as matlab_input:
        r_field = int(matlab_input.get('receptive_field')[0,0])
        n_levels = int(matlab_input.get('n_levels')[0,0])
        
        print "Reading", read_file, " receptive field:", r_field, "levels:", n_levels
        for i in range(n_levels):
            x_data = [matlab_input[element[i]][:] for element in matlab_input['inputs_formatted']] 
            ytrue_data = [matlab_input[element[i]][:] for element in matlab_input['targets']]
            x_data = x_data[0].transpose()
            ytrue_data = ytrue_data[0].transpose()
            store_file = data_location + "/xytrue_" + str(i) + "_r" + str(receptive_field) + ".pkl"
            print "Read level", i, "x:", x_data.shape, "ytrue:", ytrue_data.shape
            data_list = [ x_data, ytrue_data ]
            with open(store_file, "wb") as output_file:
                pkl.dump(data_list, output_file)

def read_data(data_location, receptive_field, level):
    store_file = data_location + "/xytrue_" + str(level) + "_r" + str(receptive_field) + ".pkl"
    try: 
        os.path.isfile(store_file)
    except IOError:
        print "Failed to load:", store_file

    with open(store_file, "rb") as input_file:
        data_list = pkl.load(input_file)
    x_data = data_list[0]
    ytrue_data = data_list[1]
    return x_data, ytrue_data
    

def read_seed(seed_file):
    try:
        os.path.isfile(seed_file)
    except IOError:
        print "Failed to load:", seed_file
    print "Opened:", seed_file

    with h5py.File(seed_file) as matlab_seed:
        seed_data = matlab_seed.get('seed')
        seed_data = np.array(seed_data)
        seed_data = seed_data.transpose()
        print "Read seed:", seed_data.shape
    return seed_data

def write_song(song, data_location, level, receptive_field):
    song_name = "song_" + str(level+1) + "_r" + str(receptive_field)
    song_file = data_location + "/" + song_name + ".mat"
    song_dict = {}
    song_dict[unicode(song_name)] = song
    hdf5storage.savemat(song_file, song_dict)
    print "Saved song:", song_file
    return song_name

