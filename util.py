import numpy as np
import os.path
import scipy.io
import h5py
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
    #matlab_input = scipy.io.loadmat(read_file)
    matlab_input = h5py.File(read_file)
    print matlab_input.keys()
    r_field = int(matlab_input['receptive_field'])
    n_levels = int(matlab_input['n_levels'])
    x_data = matlab_input['inputs_formatted']
    ytrue_data = matlab_input['targets']
    print "Reading", read_file, " receptive field:", r_field, "levels:", n_levels
    for i in range(n_levels):
        store_file = data_location + "/xytrue_" + str(i) + "_r" + str(receptive_field) + ".pkl"
        print "Read level", i, "x:", x_data[i,0].shape, "ytrue:", ytrue_data[i,0].shape
        data_list = [ x_data[i,0], ytrue_data[i,0] ]
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
    

def read_seed(seed_file, receptive_field):
    try:
        matlab_input = scipy.io.loadmat(seed_file)
    except IOError:
        print "Failed to load:", seed_file
    print "Opened:", seed_file
    seed_data = matlab_input['seed']
    print "Read seed:", seed_data.shape
    return seed_data

def write_song(song, data_location, level, receptive_field):
    song_name = "song_" + str(level+1) + "_r" + str(receptive_field)
    song_file = data_location + "/" + song_name + ".mat"
    scipy.io.savemat(song_file, mdict={ song_name: song})
    print "Saved song:", song_file
    return song_name

