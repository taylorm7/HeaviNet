import scipy.io.wavfile as wav
import numpy as np
import h5py
import hdf5storage

from filter import butter_lowpass_filter

# analog_to_digital: convert analog audio to digitally sampled audio
# inputs:
# y_nonlinear, double array of nonlinear audio
# Q, double value given by quantize step size
# outputs:
# y_digital, integer array of digitized audio
def analog_to_digital(y_nonlinear, Q):
    analog = y_nonlinear + 1
    y_digital = np.floor(analog/Q)
    # exclude any digital values out of range (-1,1)
    y_digital[y_digital >= 2/Q] = (2-Q)/Q
    y_digital[y_digital < 0] = 0
    y_digital = y_digital.astype(int)
    return y_digital

# digital_to_analog: convert digital audio to analog sampled audio
# inputs:
# y_digital, integer array of audio
# Q, double value given by quantize step size
# outputs:
# y_nonlinear, double array of analog audio
def digital_to_analog(y_digital, Q):
    analog = y_digital.astype(float)*Q
    y_nonlinear = analog -1;
    return y_nonlinear

# mu_transform: apply mu law companding to audio
# inputs:
# x, double array of raw audio
# mu, mu factor for transform
# Q, double value given by quantize step size
# outputs:
# y_nonlinear, double array of analog audio with mu law non-linear transform
def mu_trasform(x, mu, Q):
    y_nonlinear = np.sign(x)*np.log(1+mu*abs(x)) / np.log(1+mu)
    return y_nonlinear

# mu_inverse: apply inverse mu law companding to audio
# inputs:
# y_nonlinear, double array of mu law companded audio
# mu, mu factor for transform
# Q, double value given by quantize step size
# outputs:
# y, double array of analog audio with inverse mu law transform
def mu_inverse(y_nonlinear, mu, Q):
    xmax = 1;
    xmin = -1;
    y_quantized_nonlinear = np.floor((y_nonlinear-xmin)/Q)*Q+Q/2+xmin;
    y = np.sign(y_quantized_nonlinear)*(1/mu)*( np.power((1+mu),(abs(y_quantized_nonlinear)))-1 );
    return y

# raw: convert from quantized companded digital data analog un-companded audio
# inputs:
# digital, integer array of digital companded audio
# bits, number of bits for quantization
# outputs:
# y, double array of analog audio with inverse mu law transform
def raw(digital, bits=8):
    N = 2**bits
    mu = float(N-1)
    xmax = 1.0
    xmin = -1.0
    Q=(xmax-xmin)/N
    y_nonlinear = digital_to_analog(digital, Q)
    y = mu_inverse(y_nonlinear, mu, Q)
    return y

# quantize: convert from analog un-companded audio to quantized companded digital data
# inputs:
# digital, integer array of digital companded audio
# bits, number of bits for quantization
# outputs:
# y_d, integer array of digital audio with mu law transform
def quantize(analog, bits=8):
    N = 2**bits
    mu = float(N-1)
    xmax = 1.0
    xmin = -1.0
    Q=(xmax-xmin)/N
    y_nonlinear = mu_trasform(analog, mu, Q)
    y_d = analog_to_digital(y_nonlinear, Q)
    return y_d

# filter_song: filter audio song according to a frequency 
# inputs:
# song, double array of raw audio
# frequency_list, list of double values ordered according to level
# level, integer of givel level to filter
# fx, sampling frequency of song
# outputs:
# filtered_song, filtered audio according to given frequency
def filter_song(song, frequency_list, level, fx=44100):
    level_fx = frequency_list[level]
    filtered_song = butter_lowpass_filter(song, level_fx, fx)
    print("Level:", level, "Fx", level_fx)
    return filtered_song

# format_song: create a list of filtered songs for all levels
# inputs:
# song, double array of raw audio
# frequency_list, list of double values ordered according to level
# index_list, list of integer values ordered according to a levels sampling rate
# n_levels, integer of number of levels for heavinet
# data_location, string location of data directory
# bits, number of bits for quantization
# fx, sampling frequency of song
# outputs:
# song_list, filtered audio according to given frequency and fomratted into a numpy array
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

        filtered_song = mu_trasform(filtered_song, mu, Q)
        filtered_song = analog_to_digital(filtered_song, Q)        
       
        # sample filtered_song according to indicies
        indicies = np.arange(song_length)
        indicies = np.repeat(indicies, index_length)
        indicies = np.reshape(indicies, (-1,index_length))
        indicies = indicies + index_list[i]
        indicies = indicies % song_length

        song_list[i] = filtered_song[indicies]

    print("Song List", song_list.shape)
    return song_list

# format_feedval: create a list of filtered input for neural network input according to batch size
# inputs:
# song, double array of raw audio
# frequency_list, list of double values ordered according to level
# index_list, list of integer values ordered according to a levels sampling rate
# n_levels, integer of number of levels for heavinet
# data_location, string location of data directory
# bits, number of bits for quantization
# fx, sampling frequency of song
# outputs:
# song_list, filtered audio according to given frequency and fomratted into a numpy array
def format_feedval(song, frequency_list, index_list, song_length, n_levels, bits=8, fx=44100):   
    N = 2**bits
    mu = float(N-1)
    xmax = 1.0
    xmin = -1.0
    Q=(xmax-xmin)/N

    index_length = len(index_list[0])

    song_list = np.empty([n_levels, song_length, index_length], dtype=int)
    filtered_song = np.empty([song_length], dtype=float)
    for i in range(n_levels):
        filtered_song = butter_lowpass_filter(song, frequency_list[i]/2.0, fx)
        filtered_song = mu_trasform(filtered_song, mu, Q)
        filtered_song = analog_to_digital(filtered_song, Q)        
        
        indicies = len(song)-1 + index_list[i]

        song_list[i] = filtered_song[indicies]
    return song_list


