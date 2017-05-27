import tensorflow as tf
import numpy as np
import scipy.io as sio


matrix_file= sio.loadmat('/home/sable/AudioFiltering/Testing/test.mat')

mat = matrix_file['data']

print(mat[1:100,0])

print(mat.shape)


print("End")
