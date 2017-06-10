import struct
import numpy as np
import scipy.io.wavfile

def read_wav(file_location='/home/sable/HeaviNet/data/songs/clams_u.wav'):
    data = []
    header = []
    with open(file_location, "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)
        i = 0
        while ((data[i] != ord('d')) or 
                (data[i+1] != ord('a')) or 
                (data[i+2] != ord('t')) or 
                (data[i+3] != ord('a')) ):
            i = i+1
        header = data[:i+4]
        data = data[i+4:]

    #print header
    print "Data:", data, data.size
    return data, header


def write_wav(data, header, file_location='/home/sable/HeaviNet/data/songs/clams_u_out.wav'):
    out_array = []
    out_array = np.append(out_array, header.astype('uint8'))
    out_array = np.append(out_array, data.astype('uint8'))
    with open(file_location, "wb") as f:
        out_array.astype('uint8').tofile(f)
