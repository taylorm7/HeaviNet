import numpy as np
import sys
import os.path
import scipy.io
import h5py
import hdf5storage
import pickle as pkl

def load_matlab(data_location, receptive_field):
    read_file = data_location + "/matlab_song_r" + str(receptive_field) + ".mat"
    try:
        with h5py.File(read_file) as matlab_input:
            r_field = int(matlab_input.get('receptive_field')[0,0])
            n_levels = int(matlab_input.get('n_levels')[0,0])
            x_list = []
            print("Reading", read_file, " receptive field:", r_field, "levels:", n_levels)
            for i in range(n_levels):
                x_data = [matlab_input[element[i]][:] 
                        for element in matlab_input['inputs_formatted']] 
                ytrue_data = [matlab_input[element[i]][:] 
                        for element in matlab_input['targets']]
                x_data = x_data[0].transpose()
                ytrue_data = ytrue_data[0].transpose()
                store_file = data_location+"/xytrue_"+str(i)+"_r"+str(receptive_field)+".pkl"
                print("Read level", i, "x:", x_data.shape, "ytrue:", ytrue_data.shape)
                data_list = [ x_data, ytrue_data ]
                x_list.append(x_data)
                #if i == 0:
                #    x_list = x_data
                #else:
                #    x_list = np.dstack( ( x_list, x_data) )
                with open(store_file, "wb") as output_file:
                    pkl.dump(data_list, output_file, protocol=2)
            x_list = np.asarray(x_list)
            x_file = data_location+"/x_r"+str(receptive_field)+".pkl"
            print(np.shape(x_list))
            with open(x_file, "wb") as output_file:
                pkl.dump(x_list, output_file, protocol=2)
    except IOError:
        print("Failed to load:", store_file)
        sys.exit()


def read_data(data_location, receptive_field, level):
    store_file = data_location + "/xytrue_" + str(level) + "_r" + str(receptive_field) + ".pkl"
    x_file = data_location+"/x_r"+str(receptive_field)+".pkl"
    try: 
        with open(store_file, "rb") as input_file:
            data_list = pkl.load(input_file)
            x_data = data_list[0]
            ytrue_data = data_list[1]
        with open(x_file, "rb") as input_file:
            x_list = pkl.load(input_file)
        print(np.shape(x_data), np.shape(ytrue_data), np.shape(x_list))
        return x_data, ytrue_data, x_list
    except IOError:
        print("Failed to load:", store_file)
        sys.exit() 

def read_seed(seed_file):
    try:
        with h5py.File(seed_file) as matlab_seed:
            seed_data = matlab_seed.get('seed')
            seed_data = np.array(seed_data)
            seed_data = seed_data.transpose()
            print("Read seed:", seed_data.shape)
            return seed_data
    except IOError:
        print("Failed to load:", seed_file)
        sys.exit()

def write_song(song, data_location, level, receptive_field):
    song_name = "song_" + str(level+1) + "_r" + str(receptive_field)
    song_file = data_location + "/" + song_name + ".mat"
    song_dict = {}
    song_dict[str(song_name)] = song
    hdf5storage.savemat(song_file, song_dict)
    print("Saved song:", song_file)
    return song_name

