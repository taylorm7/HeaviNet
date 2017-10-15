import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

from filter import butter_lowpass_filter

fx, song = wav.read('/home/sable/HeaviNet/data/songs/bach_10.wav')
song = song.astype(float)
print(fx, song.shape)
s = song.sum(axis=1) / 2
s = s/abs(s).max()

out = np.int16(s/np.max(np.abs(s)) * 32767 )
wav.write('s.wav', fx, out)

s_1 = butter_lowpass_filter(s, 500, fx)
out_1 = np.int16(s_1/np.max(np.abs(s_1)) * 32767 )
wav.write('s_1.wav', fx, s_1)

plt.plot(s)
plt.plot(s_1)
plt.show()
