import numpy as np

def format_level(iterable, receptive_field):
    print type(iterable)
    level = []
    for i in range(len(iterable)):
        l = []
        for j in range(-receptive_field, 1):
            index = ( i - 2**abs(j) )%len(iterable)
            l = np.append(l, iterable[index])
        l = np.append(l, iterable[i])
        for j in range( receptive_field +1 ):
            index = ( i + 2**abs(j) )%len(iterable)
            l = np.append(l, iterable[index])
        level = np.append(level, l)
    print level.shape
    print level[0,:]
