import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

from filter import butter_lowpass_filter

#bits = 8
#N = 2**bits
#mu = float(N-1)
#xmax = 1.0
#xmin = -1.0
#Q=(xmax-xmin)/N

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
