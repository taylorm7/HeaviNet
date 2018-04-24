import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt

# butter_lowpass: create lowpass filter
# inputs:
# cutoff, double cutoff frequency for filter
# fs, double sampling rate for filter
# order, integer filter order
# outputs:
# b,a doulbe filter values for use in lfilter function
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
# butter_lowpass_filter: filter data according to cuttoff frequency and sampling rate
# inputs:
# data, double array of audio data to be filtered
# cutoff, double value for frequency of filter
# fx, double value for audio sampling rate
# order, integer value for filter order
# outputs:
# y, double array of filtered audio
def butter_lowpass_filter(data, cutoff, fx, order=5):
    if cutoff >= fx/2.0:
        return data
    b, a = butter_lowpass(cutoff, fx, order=order)
    y = lfilter(b, a, data)
    return y


