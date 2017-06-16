import tensorflow as tf
import numpy as np

import scipy.io as sio
from util import format_level

matlab_input = sio.loadmat('/home/sable/HeaviNet/data/matlabSong2heavinet.mat')

x_1 = format_level(matlab_input['level_1'], 12)

