import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from random import randint

clip_size = 32*4*120
n_clips = 1

num_classes = 256

batch_size = 128

rnn_size = 128


def make_onehot(onehot_values, onehot_classes):
    onehot_matrix = np.zeros((onehot_values.size, onehot_classes))
    onehot_matrix[np.arange(onehot_values.size),onehot_values] = 1
    return onehot_matrix

def batch(iterable, start, batches=0, size=clip_size):
    if start + size + 1 > len(iterable):
        return np.zeros((0, size)), np.zeros((0 , num_classes)), np.zeros(0)
    b_clip = iterable[start:start+size, :]
    b_y = iterable[start+size, 0]
    for i in range(1,batches):
        if start+size+i+1 <= len(iterable):
            b_clip = np.append(b_clip, iterable[start+i:start+size+i, :], axis = 0) 
            b_y = np.append(b_y, iterable[start+i+size, 0])

    batch_overflow = len(b_clip) % size
    if batch_overflow != 0:
        b_clip =  b_clip[:-batch_overflow or None, :]
    b_clip = b_clip.reshape( (-1, size) )    
    b_onehot = make_onehot(b_y, num_classes)
    return b_clip, b_onehot, b_y 

def batch_r3(iterable, start, batches):
    b_clip = np.zeros((0, n_clips, clip_size))
    b_onehot = np.zeros((0 , n_clips,  num_classes))
    b_y =np.zeros((0, n_clips))
    
    if start + clip_size + batches - 1 > len(iterable):
        return b_clip, b_onehot, b_y 
        
    for i in range(batches):
        clip, onehot, y = batch(iterable, start, n_clips)
        b_clip = np.append(b_clip, clip.reshape((1,n_clips, clip_size)), axis = 0 )
        b_onehot = np.append(b_onehot, onehot.reshape((1, n_clips, num_classes)),axis=0)
        b_y = np.append(b_y, y.reshape((1,n_clips)), axis=0)
    return b_clip, b_onehot, b_y 
    
    


matrix_file= sio.loadmat('/home/sable/AudioFiltering/Testing/test.mat')
mat = matrix_file['data']
print type(mat)
#print(matrix_file)
#print(mat[1:100,0], mat.shape)

# input vector
x = tf.placeholder(tf.float32, [None, n_clips])
x_d3 = tf.reshape(x, (-1, n_clips, 1))
x_onehot = tf.one_hot(tf.cast(x,tf.int32), num_classes)
print x_onehot.shape
print x_d3
y_true = tf.placeholder(tf.float32)
y_true_cls = tf.placeholder(tf.int64)

def neural_network_model(data):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, num_classes])),
             'biases':tf.Variable(tf.random_normal([num_classes]))}
    #x = tf.transpose(data, [-1,0,2])
    data = tf.reshape(data, [-1, n_clips])
    data = tf.split(data, n_clips, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, data, dtype=tf.float32)
    print "rnn outputs", outputs
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
    print "output", output
    return output

#weights = tf.Variable(tf.zeros([clip_size, num_classes]))
#biases = tf.Variable(tf.zeros([num_classes]))
#logits = tf.matmul(x, weights) + biases
logits = neural_network_model(x)
print "logits", logits
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
#optimizer = tf.train.AdamOptimizer().minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("x shape", x.shape)
print("y predicted shape", y_pred.shape)
print("y predicted class shape", y_pred_cls.shape)
print("logits shape", logits.shape)
print("cross entropy shape", cross_entropy.shape)
print("cost shape", cost.shape)

session = tf.Session()
session.run(tf.global_variables_initializer())

def print_accuracy(feed_dict_test):
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

def optimize(epochs, iterations=len(mat) ):
    for i in range(epochs):
        start = 0
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        for start in range(0, iterations, batch_size):
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            #x_batch, y_true_batch = data.train.next_batch(batch_size)
            x_batch, y_onehot_batch, y_class_batch = batch(mat,start, batch_size, n_clips)
            
            print(start, "x batch", x_batch.shape, "y batch", y_onehot_batch.shape, type(x_batch), type(y_onehot_batch) )
            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            # Note that the placeholder for y_true_cls is not set
            # because it is not used during training.
            feed_dict_train = {x: x_batch,
                               y_true: y_onehot_batch,
                               y_true_cls: y_class_batch }
            
            #print x_batch.shape, "x batch\n", x_batch
            #print y_onehot_batch.shape, "y true batch\n", y_onehot_batch
            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            o, c, cor, y = session.run([optimizer, cost, correct_prediction, y_pred_cls], feed_dict=feed_dict_train)
            epoch_correct += np.sum(cor)
            epoch_total += cor.size
            epoch_loss += c
            print y.shape, "y predicted", y
            #print correct_prediction
        print epoch_total,":epoch", i, "completed w/ loss", epoch_loss, "correct", epoch_correct, "and percentage" , 100.0 * float(epoch_correct) / float(epoch_total)

        #test_index = randint(0, len(mat)-1000)
        #print "test at", test_index
        #test_clip, test_y_onehot, test_y = batch(mat, test_index, 10000)
        #feed_dict_test = {x: test_clip, y_true: test_y_onehot, y_true_cls: test_y} 
        #print_accuracy(feed_dict_test)


def predict():
    predictions = session.run(y_pred_cls, feed_dict=feed_dict_test)
    return predictions

def create(song_seed, length):
    song_index = len(song_seed)
    for i in range(length):
        song_new_y = session.run(y_pred_cls, 
                                {x: song_seed[song_index-clip_size:song_index].reshape( (1,clip_size)) } )
        song_seed = np.append(song_seed, song_new_y, axis = 0)
    return song_seed

#song, _, _ = batch(mat, 20000, 1)
#song = song.reshape( (clip_size) )

optimize(3, batch_size)
#song = create(song, 100)
#print song, song.shape


'''
x_batch, y_onehot_batch, y_class_batch = batch_r3(mat,10, batch_size)
            
print x_batch.shape, "x batch\n", x_batch
print y_onehot_batch.shape, "y onehot batch/n", y_onehot_batch
print y_class_batch.shape, "y class batch\n", y_class_batch
'''
session.close()
