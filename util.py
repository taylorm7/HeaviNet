import numpy as np
import os.path
import scipy.io
import cPickle as pkl

def name_level(n_levels):
    x_name = []
    ytrue_name = []
    for i in range(1,n_levels+1):
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

def read_data(receptive_field, level_names, ytrue_names, force_read=False):

    x_data = {}
    ytrue_data = {}

    if os.path.isfile("data/x_ytrue.pkl" ) and force_read==False :
        print "Pickle Load file x_ytrue.pkl"
        with open(r"data/x_ytrue.pkl", "rb") as input_file:
            data_list = pkl.load(input_file)
        x_data = data_list[0]
        ytrue_data = data_list[1]
    else:
        print "Matlab Load file matlabSong2heavinet.mat" 
        matlab_input = scipy.io.loadmat("data/matlabSong2heavinet.mat")
        for name in level_names:
            x_data[name] = format_level(matlab_input[name], receptive_field)   
        for name in ytrue_names:
            ytrue_data[name] = matlab_input[name]
        
        data_list = [x_data, ytrue_data]
        
        with open(r"data/x_ytrue.pkl", "wb") as output_file:
            pkl.dump(data_list, output_file)
    return x_data, ytrue_data
