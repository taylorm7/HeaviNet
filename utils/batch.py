import numpy as np

def make_onehot(onehot_values, onehot_classes):
    onehot_matrix = np.zeros((onehot_values.size, onehot_classes))
    onehot_matrix[np.arange(onehot_values.size),onehot_values] = 1
    return onehot_matrix

def make_batch(iterable, start, batches, clip_size):
    
    list_ = ['a','b','c','d','e','f','g','h']
    n = 3  # group size
    m = 1  # overlap size
    #[list_[i:i+n-m+1] for i in xrange(0,len(list_), n-m)]
    print len(iterable) -2
    batch = [iterable[i:i+clip_size] for i in xrange(start, 
        min(start+batches, len(iterable)-(clip_size) ) , 1) ]
    print len(batch)
    batch = np.reshape(batch, (len(batch), clip_size))
    print batch

def batch(iterable, start, batches, clip_size, n_classes):
    if start + clip_size + 1 > len(iterable):
        return np.zeros((0, clip_size)), np.zeros((0 , n_classes)), np.zeros(0)
    b_clip = iterable[start:start+clip_size, :]
    b_y = iterable[start+clip_size, 0]
    for i in range(1,batches):
        if start+clip_size+i+1 <= len(iterable):
            b_clip = np.append(b_clip, iterable[start+i:start+clip_size+i, :], axis = 0) 
            b_y = np.append(b_y, iterable[start+i+clip_size, 0])

    batch_overflow = len(b_clip) % clip_size
    if batch_overflow != 0:
        b_clip =  b_clip[:-batch_overflow or None, :]
    b_clip = b_clip.reshape( (-1, clip_size) )    
    b_onehot = make_onehot(b_y, n_classes)
    return b_clip, b_onehot, b_y 

def batch_r3(iterable, start, batches, n_clips, clip_size, n_classes):
    b_clip = np.zeros((0, n_clips, clip_size))
    b_onehot = np.zeros((0 , n_clips,  n_classes))
    b_y =np.zeros((0, n_clips))
    
    if start + clip_size + batches - 1 > len(iterable):
        return b_clip, b_onehot, b_y 
        
    for i in range(batches):
        clip, onehot, y = batch(iterable, start, n_clips)
        b_clip = np.append(b_clip, clip.reshape((1,n_clips, clip_size)), axis = 0 )
        b_onehot = np.append(b_onehot, onehot.reshape((1, n_clips, n_classes)),axis=0)
        b_y = np.append(b_y, y.reshape((1,n_clips)), axis=0)
    return b_clip, b_onehot, b_y 

