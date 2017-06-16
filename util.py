import numpy as np
import os.path
import scipy.io

def format_level(iterable, receptive_field):
    print type(iterable)
    
    indicies = range(receptive_field, -1, -1)
    for i in range(len(indicies)):
        indicies[i] = -1* 2**indicies[i]
    indicies.append(0)
    indicies_pos = range(0, receptive_field+1)
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

def read_data(data_path, force_read=False):

    data = []

    if os.path.isfile(data_path + "matlab_input.npy" ):
        print "PYC Load file:" + data_path + "matlab_input.npy"
        data = np.load(data_path + "matlab_input.npy")
    else:
        print "Matlab Load file" + data_path +"matlabSong2heavinet.mat" 
        matlab_input = scipy.io.loadmat(data_path + "matlabSong2heavinet.mat")
        data = np.append(data, format_level(matlab_input['level_1'], 5))
        data = np.append(data, format_level(matlab_input['level_2'], 5))
    print data.shape
