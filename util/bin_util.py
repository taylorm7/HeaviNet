import struct
import numpy as np
import scipy.io.wavfile

data = []
header = []
with open("/home/sable/HeaviNet/data/songs/ghost_u.wav", "rb") as f:
    data = np.fromfile(f, dtype=np.uint8)
    print data
    i = 0
    while ((data[i] != ord('d')) or 
            (data[i+1] != ord('a')) or 
            (data[i+2] != ord('t')) or 
            (data[i+3] != ord('a')) ):
        print data[i]
        i = i+1
    print data[i], data[i+1], data[i+2], data[i+3], i
    header = data[:i+4]
    data = data[i+4:]
    print data.size
    print data[0], data[1], data[2], data[3]

print header
print data

out_array = []
out_array = np.append(out_array, header)
out_array = np.append(out_array, data)


out_array.astype('uint8').tofile("/home/sable/HeaviNet/data/songs/ghost_u_out.wav")

#with open("/home/sable/HeaviNet/data/songs/clams_u_out.wav", "wb") as f:
    



#scipy.io.wavfile.write('/home/sable/HeaviNet/data/songs/clams_u_out.wav', 8000, data)
