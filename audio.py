import scipy.io.wavfile as wav
import numpy as np
import h5py
import hdf5storage

from filter import butter_lowpass_filter, savitzky_golay

#bits = 8
#N = 2**bits
#mu = float(N-1)
#xmax = 1.0
#xmin = -1.0
#Q=(xmax-xmin)/N

# hard coded levels; derived from 8 levels with piano music
levels = [ 5.0, 5.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]

def analog_to_digital(y_nonlinear, Q):
    analog = y_nonlinear + 1
    y_digital = np.floor(analog/Q)
    # exclude any digital values out of range (-1,1)
    y_digital[y_digital >= 2/Q] = (2-Q)/Q
    y_digital[y_digital < 0] = 0
    y_digital = y_digital.astype(int)
    return y_digital

def digital_to_analog(y_digital, Q):
    analog = y_digital.astype(float)*Q
    y_nonlinear = analog -1;
    return y_nonlinear

def mu_trasform(x, mu, Q):
    y_nonlinear = np.sign(x)*np.log(1+mu*abs(x)) / np.log(1+mu)
    return y_nonlinear

def mu_inverse(y_nonlinear, mu, Q):
    xmax = 1;
    xmin = -1;
    y_quantized_nonlinear = np.floor((y_nonlinear-xmin)/Q)*Q+Q/2+xmin;
    y = np.sign(y_quantized_nonlinear)*(1/mu)*( np.power((1+mu),(abs(y_quantized_nonlinear)))-1 );
    return y

def raw(digital, bits=8):
    N = 2**bits
    mu = float(N-1)
    xmax = 1.0
    xmin = -1.0
    Q=(xmax-xmin)/N
    y_nonlinear = digital_to_analog(digital, Q)
    y = mu_inverse(y_nonlinear, mu, Q)
    return y

def quantize(analog, bits=8):
    N = 2**bits
    mu = float(N-1)
    xmax = 1.0
    xmin = -1.0
    Q=(xmax-xmin)/N
    y_nonlinear = mu_trasform(analog, mu, Q)
    y_d = analog_to_digital(y_nonlinear, Q)
    return y_d

def test_write(song, level, data_location ):
    print("Out Song:", level)
    digital = quantize(song)
    out = raw(digital)

    song_name = "s" + str(level)
    song_file = data_location + "/" + song_name + ".mat"
    song_dict = {}
    song_dict[str(song_name)] = song
    hdf5storage.savemat(song_file, song_dict)
    print("Saved song:", song_file)

    song_name = "d" + str(level)
    song_file = data_location + "/" + song_name + ".mat"
    song_dict = {}
    song_dict[str(song_name)] = digital
    hdf5storage.savemat(song_file, song_dict)
    print("Saved song:", song_file)

    target_name = "t" + str(level)
    target_file = data_location + "/" + target_name + ".mat"
    target_dict = {}
    target_dict[str(target_name)] = out
    hdf5storage.savemat(target_file, target_dict)
    print("Saved target:", target_file)


def test_songs(song, frequency_list, n_levels, data_location, fx=44100):
    for i in range(n_levels):
        l_fx = frequency_list[i]/2.0
        print("L:", i, "Fx", l_fx)
        filtered = butter_lowpass_filter(song, l_fx, fx)
        test_write(filtered, i, data_location)


def format_song(song, frequency_list, index_list, song_length, n_levels, data_location, bits=8, fx=44100):   
    N = 2**bits
    mu = float(N-1)
    xmax = 1.0
    xmin = -1.0
    Q=(xmax-xmin)/N
    
    index_length = len(index_list[0])
    song_length = len(song)

    song_list = np.empty([n_levels, song_length, index_length], dtype=float)
    filtered_song = np.empty([song_length])
    print("Song:", song_length, "Index", index_length, "Song List", song_list.shape)
    for i in range(n_levels):
        level_fx = frequency_list[i]/2.0;
        filtered_song = butter_lowpass_filter(song, level_fx, fx)

        #song_max = max(abs(filtered_song))
        #print("Max at", song_max)
        #print("Rate:", 1.0/song_max)

        #filtered_song = filtered_song * levels[i]
        #test_write(filtered_song, i , data_location)

        filtered_song = mu_trasform(filtered_song, mu, Q)
        filtered_song = analog_to_digital(filtered_song, Q)        
        #print("Level", i,"FX", level_fx)
        #print("Filtered song", filtered_song.shape)
        #print(filtered_song)
        
        indicies = np.arange(song_length)
        indicies = np.repeat(indicies, index_length)
        indicies = np.reshape(indicies, (-1,index_length))
        indicies = indicies + index_list[i]
        #print("indicies", indicies.shape)
        #print(indicies)
        indicies = indicies % song_length
        #print(indicies)

        song_list[i] = filtered_song[indicies]

    print("Song List", song_list.shape)
    return song_list

def format_feedval(song, frequency_list, index_list, song_length, n_levels, bits=8, fx=44100):   
    N = 2**bits
    mu = float(N-1)
    xmax = 1.0
    xmin = -1.0
    Q=(xmax-xmin)/N

    index_length = len(index_list[0])

    song_list = np.empty([n_levels, song_length, index_length], dtype=int)
    filtered_song = np.empty([song_length], dtype=float)
    #song = savitzky_golay(song, 41, 5)
    #print("Song:", song_length, "Index", index_length, "Song List", song_list.shape)
    #print(song[ len(song)-1], song[len(song)-2] )
    for i in range(n_levels):
        filtered_song = butter_lowpass_filter(song, frequency_list[i]/2.0, fx)
        filtered_song = mu_trasform(filtered_song, mu, Q)
        filtered_song = analog_to_digital(filtered_song, Q)        
        #print("Filtered song", filtered_song.shape)
        #print(filtered_song)
        
        
        #indicies = np.arange(song_length)
        #indicies = np.repeat(indicies, index_length)
        #indicies = np.reshape(indicies, (-1,index_length))
        indicies = len(song)-1 + index_list[i]
        
        #print("indicies", indicies.shape)
        #print(indicies)

        song_list[i] = filtered_song[indicies]
    #print("Song List", song_list.shape)
    return song_list

'''
print(bits, N, mu, Q)

fx, song = wav.read('/home/sable/HeaviNet/data/songs/bach_10.wav')
song = song.astype(float)
print(fx, song.shape)
s = song.sum(axis=1) / 2
s = s/abs(s).max()

#out = np.int16(s/np.max(np.abs(s)) * 32767 )
wav.write('s.wav', fx, s)

y_nonlinear = mu_trasform(s, mu, Q)
y_d = analog_to_digital(y_nonlinear, Q)
y_a = digital_to_analog(y_d, Q)
y = mu_inverse(y_a, mu, Q)

wav.write('s_out.wav', fx, y)

s_1 = butter_lowpass_filter(s, 500, fx)
out_1 = np.int16(s_1/np.max(np.abs(s_1)) * 32767 )
wav.write('s_1.wav', fx, s_1)

#plt.plot(s)
#plt.plot(s_1)
#plt.show()
'''
