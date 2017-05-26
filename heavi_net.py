import tensorflow as tf
import numpy as np


mat = np.loadtxt(open("test.csv", "rb"), delimiter=",", skiprows=1)

print (mat)
print(mat.shape)

print("End")
