import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


matrix_file= sio.loadmat('/home/sable/AudioFiltering/Testing/test.mat')

mat = matrix_file['data']

print(matrix_file)
print(mat.shape)
print(mat[1:100,0])

clip_size = 7
num_classes = 21
batch_size = 100

t_mat = np.matrix('1 ; 2 ; 3 ; 4 ;5 ;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20')


def make_onehot(onehot_values, onehot_classes):
    onehot_matrix = np.zeros((onehot_values.size, onehot_classes))
    onehot_matrix[np.arange(onehot_values.size),onehot_values] = 1
    return onehot_matrix

def batch(iterable, start, batches=0):
    if start + clip_size + 1 > len(iterable):
        return np.zeros((0, clip_size)), np.zeros((0 , num_classes))
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
    return b_clip, b_y, b_onehot


batch_clip, batch_y, batch_y_onehot = batch(t_mat,0, 50)

print(batch_clip.shape)
print(batch_clip)
print(batch_y.shape)
print(batch_y)
print(batch_y_onehot.shape)
print(batch_y_onehot)



t_y =  tf.one_hot(batch_y, num_classes)

quit()
'''
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
v = sess.run(t_y)    
print v 

sess.close()
'''
print

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

#create set of single values for data.test
data.test.cls = np.array([label.argmax() for label in data.test.labels])

print(mat.shape)
print(data.test.images.shape)
print(data.test.labels.shape)
print(data.test.cls.shape)


# input vector
x = tf.placeholder(tf.float32, [None, clip_size])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

weights = tf.Variable(tf.zeros([clip_size, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

logits = tf.matmul(x, weights) + biases

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

print("logits shape", logits.shape)
print("y predicted shape", y_pred.shape)
print("y predicted class shape", y_pred_cls.shape)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

print("cross entropy shape", cross_entropy.shape)
print("cost shape", cost.shape)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()


session.run(tf.global_variables_initializer())



def optimize(num_iterations):
    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        print("x batch", x_batch.shape, "y batch", y_true_batch.shape, type(x_batch), type(y_true_batch) )
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)


feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}

print("x test", data.test.images.shape , "y true", data.test.labels.shape, "y true cls", data.test.cls.shape)

def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

optimize(num_iterations=1)

print_accuracy()

optimize(num_iterations=9)

print_accuracy()

#optimize(num_iterations=990)

print_accuracy()


session.close()
