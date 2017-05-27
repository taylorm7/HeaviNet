import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


matrix_file= sio.loadmat('/home/sable/AudioFiltering/Testing/test.mat')

mat = matrix_file['data']

print(mat[1:100,0])

print(mat.shape)


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

#create set of single values for data.test
data.test.cls = np.array([label.argmax() for label in data.test.labels])

print(data.test.labels.shape)
print(data.test.cls.shape)
