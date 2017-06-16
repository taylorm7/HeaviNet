import tensorflow as tf
import numpy as np
import scipy.io as sio

from util import read_data, name_level, format_level


receptive_field = 5
n_levels = 7

level_names, ytrue_names = name_level(n_levels)

print level_names, ytrue_names

x_data, ytrue_data = read_data(receptive_field, level_names, ytrue_names, force_read=True)

print x_data['level_4'].size, ytrue_data['ytrue_4'].size
