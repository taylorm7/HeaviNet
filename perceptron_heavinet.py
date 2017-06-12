import tensorflow as tf
import numpy as np
import scipy.io as sio
import scipy.io.wavfile as siow
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
from random import randint

clip_size = 32

num_classes = 256
batch_size = 128

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

def make_onehot(onehot_values, onehot_classes):
    onehot_matrix = np.zeros((onehot_values.size, onehot_classes))
    onehot_matrix[np.arange(onehot_values.size),onehot_values] = 1
    return onehot_matrix

def batch(iterable, start, batches=0):
    if start + clip_size + 1 > len(iterable):
        return np.zeros((0, clip_size)), np.zeros((0 , num_classes)), np.zeros(0)
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
    b_onehot = make_onehot(b_y, num_classes)
    return b_clip, b_onehot, b_y 

matrix_file= sio.loadmat('/home/sable/HeaviNet/data/input.mat')
mat = matrix_file['data']
print type(mat)
#print(matrix_file)
print(mat[1:100,0], mat.shape)

# input vector
x = tf.placeholder(tf.float32, [None, clip_size])
x_onehot = tf.one_hot(tf.cast(x,tf.int32), num_classes)
x_onehot = tf.reshape(x_onehot, (-1, clip_size*num_classes))
x_onehot = tf.cast(x_onehot, tf.float32)
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([num_classes*clip_size, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, num_classes])),
                    'biases':tf.Variable(tf.random_normal([num_classes]))}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.tanh(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.tanh(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.tanh(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    return output

weights = tf.Variable(tf.random_normal([num_classes, num_classes]))
biases = tf.Variable(tf.random_normal([num_classes]))
#logits = tf.matmul(x_onehot, weights) + biases
logits = neural_network_model(x_onehot)

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
optimizer = tf.train.AdamOptimizer().minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("x shape", x.shape)
print("x_one shape", x_onehot.shape)
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
            x_batch, y_onehot_batch, y_class_batch = batch(mat,start, batch_size)
            
            #print(start, "x batch", x_batch.shape, "y batch", y_onehot_batch.shape, type(x_batch), type(y_onehot_batch) )
            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            # Note that the placeholder for y_true_cls is not set
            # because it is not used during training.
            feed_dict_train = {x: x_batch,
                               y_true: y_onehot_batch,
                               y_true_cls: y_class_batch }

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            o, c, cor, y = session.run([optimizer, cost, correct_prediction, y_pred_cls], feed_dict=feed_dict_train)
            #print y
            epoch_correct += np.sum(cor)
            epoch_total += cor.size
            epoch_loss += c
        print epoch_total,":epoch", i, "completed w/ loss", epoch_loss, "correct", epoch_correct, "and percentage" , 100.0 * float(epoch_correct) / float(epoch_total)
        #test_index = randint(0, len(mat)-1000)
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

song, _, _ = batch(mat, 20000, 1)
song = song.reshape( (clip_size) )

optimize(1, 10000)
song = create(song, 100)
song = np.reshape(song, (-1,1)).astype(np.uint8)
print song, song.shape
sio.savemat('/home/sable/HeaviNet/data/out.mat', mdict={'song_out': song})

print mat.size
siow.write('/home/sable/HeaviNet/data/song.wav', 8000, mat)

session.close()