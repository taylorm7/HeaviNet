import numpy as np
import sys
import os.path
import scipy.io
import h5py
import hdf5storage
import pickle as pkl

def load_matlab(data_location, receptive_field, training_data = True):
    if training_data == True:
        read_file = data_location + "/matlab_song_r" + str(receptive_field) + ".mat"
        x_file = data_location+"/x_r"+str(receptive_field)+".pkl"
        store_file = data_location+"/xytrue_"
    else:
        read_file = data_location + "/matlab_seed_r" + str(receptive_field) + ".mat"
        x_file = data_location+"/x_r"+str(receptive_field)+".pkl"
        store_file = data_location+"/seed_"
    try:
        with h5py.File(read_file) as matlab_input:
            fx = int(matlab_input.get('fx')[0,0])
            r_field = int(matlab_input.get('receptive_field')[0,0])
            n_levels = int(matlab_input.get('n_levels')[0,0])
            song_fx = int(matlab_input.get('fx')[0,0])
            song = matlab_input.get('song')
            song = song[0]
            x_list = []
            index_list = []
            frequency_list = np.zeros(n_levels)
            print("Reading", read_file, "song", song.shape, "receptive field:", r_field, "levels:", n_levels, "fx", song_fx)

            for i in range(n_levels):
                '''
                x_data = [matlab_input[element[i]][:] 
                        for element in matlab_input['inputs_formatted']] 
                ytrue_data = [matlab_input[element[i]][:] 
                        for element in matlab_input['targets']]

                x_data = x_data[0].transpose()
                ytrue_data = ytrue_data[0].transpose()
                
                data_list = [ x_data, ytrue_data ]
                x_list.append(x_data)
                '''

                level_index = [matlab_input[element[i]][:] 
                        for element in matlab_input['indicies']]
                fx = [matlab_input[element[i]][:] 
                        for element in matlab_input['frequencies']]
                level_factor = [matlab_input[element[i]][:] 
                        for element in matlab_input['factors']]
                
                level_index = level_index[0].flatten()
                fx = fx[0][0,0]
                level_factor = level_factor[0][0,0]


                sf = store_file + str(i)+"_r"+str(receptive_field)+".pkl"
                index_list.append(level_index)
                frequency_list[i] = fx
                factor_list[i] = level_factor
                print("Read level", i, "Frequency", fx, "Factor", level_factor, "Index", level_index)
                #with open(sf, "wb") as output_file:
                #    pkl.dump(data_list, output_file, protocol=4)
            
            #x_list = np.asarray(x_list)
            index_list = np.asarray(index_list)
            index_list = index_list.astype(int)
            #print(np.shape(x_list))
            print("Index List", np.shape(index_list))
            print("Frequencies", frequency_list)
            with open(x_file, "wb") as output_file:
                pkl.dump([index_list, frequency_list, song, song_fx] , output_file, protocol=4)
    except IOError:
        print("Failed to load:", store_file)
        sys.exit()

def read_index(data_location, receptive_field):
    x_file = data_location+"/x_r"+str(receptive_field)+".pkl"
    try:
        with open(x_file, "rb") as input_file:
            _list = pkl.load(input_file)
            index_list = _list[0]
            frequency_list = _list[1]
            song = _list[2]
            song_fx = _list[3]
        print("Read index:", np.shape(index_list), np.shape(frequency_list), np.shape(song))
        return index_list, frequency_list, song, song_fx
    except IOError:
        print("Failed to load:", x_file)
        sys.exit() 

def read_data(data_location, receptive_field, level, training_data = True):
    if training_data == True:
        store_file = data_location + "/xytrue_" + str(level) + "_r" + str(receptive_field) + ".pkl"
        x_file = data_location+"/x_r"+str(receptive_field)+".pkl"
    else:
        store_file = data_location + "/seed_" + str(level) + "_r" + str(receptive_field) + ".pkl"
        x_file = data_location+"/seed_x_r"+str(receptive_field)+".pkl"
    try: 
        with open(store_file, "rb") as input_file:
            data_list = pkl.load(input_file)
            x_data = data_list[0]
            ytrue_data = data_list[1]
        with open(x_file, "rb") as input_file:
            _list = pkl.load(input_file)
            x_list = _list[0]
        print("Read data:", np.shape(x_data), np.shape(ytrue_data), np.shape(x_list))
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

def write_song(song, fx, data_location, level, receptive_field, epochs):
    seed_name = os.path.split(data_location)[1].split('.')[0]
    training_name = (os.path.split( os.path.split(data_location)[0])[1]).split('.')[0]

    song_name = "l" + str(level) + "_r" + str(receptive_field) + "_" + str(epochs) + "_" + training_name + "_" + seed_name 
    song_file = data_location + "/" + song_name + ".mat"
    song_dict = {}
    song_dict[str(song_name)] = song
    song_dict['fx'] = float(fx)
    hdf5storage.savemat(song_file, song_dict)
    print("Saved song:", song_file)
    return song_name

