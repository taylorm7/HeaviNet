import numpy as np
import os.path
import scipy.io
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

def read_data(receptive_field, level_names, ytrue_names, data_location, force_read=False):
    x_data = {}
    ytrue_data = {}
    
    read_file = data_location + "/matlab_song.mat"
    store_file = data_location + "/x_ytrue_r" + str(receptive_field) + ".pkl"

    if os.path.isfile(store_file) and force_read==False :
        print "Pickle load:", store_file
        with open(store_file, "rb") as input_file:
            data_list = pkl.load(input_file)
        x_data = data_list[0]
        ytrue_data = data_list[1]
    else:
        print "Matlab load:", read_file
        matlab_input = scipy.io.loadmat(read_file)
        for name in level_names:
            x_data[name] = format_level(matlab_input[name], receptive_field)   
        for name in ytrue_names:
            ytrue_data[name] = matlab_input[name]
        
        data_list = [x_data, ytrue_data]
        
        with open(store_file, "wb") as output_file:
            pkl.dump(data_list, output_file)
    return x_data, ytrue_data
def load_matlab(data_location, receptive_field):
    read_file = data_location + "/matlab_song.mat"
    matlab_input = scipy.io.loadmat(read_file)
    n_levels = int(matlab_input['n_levels'])
    x_names, ytrue_names = name_level(n_levels)
    
    x_data = {}
    ytrue_data = {}
    for xn, ytn in zip(x_names, ytrue_names):
        x_data[xn] = format_level(matlab_input[xn], receptive_field)   
        ytrue_data[ytn] = matlab_input[ytn]
        print "Read", xn ,"x:" , x_data[xn].size, "ytrue:", ytrue_data[ytn].size

    for xn, ytn, i in zip(x_names, ytrue_names, range(n_levels)):
        store_file = data_location + "/xytrue_" + str(i) + "_r" + str(receptive_field) + ".pkl"
        data_list = [x_data[xn], ytrue_data[ytn]]
        with open(store_file, "wb") as output_file:
            pkl.dump(data_list, output_file)
        print "Stored:", store_file

def read_2(data_location, receptive_field, level):
    store_file = data_location + "/xytrue_" + str(level) + "_r" + str(receptive_field) + ".pkl"
    if os.path.isfile(store_file):
        with open(store_file, "rb") as input_file:
            data_list = pkl.load(input_file)
        x_data = data_list[0]
        ytrue_data = data_list[1]
        return x_data, ytrue_data
    else:
        print "Failed to load:", store_file
    

def read_seed(seed_file, receptive_field):
    matlab_input = scipy.io.loadmat(seed_file)
    seed_data = matlab_input['seed']
    formated_seed_data = format_level(seed_data,receptive_field)
    print "Read seed:", seed_data.size, " formatted:", formated_seed_data.size
    return formated_seed_data

def write_song(song, data_location, level, receptive_field):
    song_name = "song_" + str(level) + "_r" + str(receptive_field)
    song_file = data_location + "/" + song_name + ".mat"
    scipy.io.savemat(song_file, mdict={ song_name: song})
    print "Saved song:", song_file
    return song_name

