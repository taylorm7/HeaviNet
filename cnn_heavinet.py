import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
from random import randint

clip_size = 4
clip_image = 2

num_classes = 256
batch_size = 512


num_channels = 1
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

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

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


#### Helper-function for creating a new Fully-Connected Layer

# This function creates a new fully-connected layer in the computational graph for TensorFlow. Nothing is actually calculated here, we are just adding the mathematical formulas to the TensorFlow graph.
# 
# It is assumed that the input is a 2-dim tensor of shape `[num_images, num_inputs]`. The output is a 2-dim tensor of shape `[num_images, num_outputs]`.

# In[16]:

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

matrix_file= sio.loadmat('/home/sable/AudioFiltering/Testing/test.mat')
mat = matrix_file['data']
print type(mat)
#print(matrix_file)
#print(mat[1:100,0], mat.shape)

# input vector
x = tf.placeholder(tf.float32, [None, clip_size])
x_onehot = tf.one_hot(tf.cast(x,tf.int32), num_classes)
x_onehot = tf.reshape(x_onehot, (-1, clip_size*num_classes))
x_onehot = tf.cast(x_onehot, tf.float32)

x_image = tf.reshape(x, [-1, clip_image, clip_image, num_channels])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

def neural_network_model(data):
    layer_conv1, weights_conv1 = new_conv_layer(input=data, 
        num_input_channels=num_channels,
        filter_size=filter_size1,
        num_filters=num_filters1,
        use_pooling=False)
    layer_conv1

    layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
        num_input_channels=num_filters1,
        filter_size=filter_size2,
        num_filters=num_filters2,
        use_pooling=False)
    layer_conv2

    layer_flat, num_features = flatten_layer(layer_conv2)
    layer_flat
    num_features

    layer_fc1 = new_fc_layer(input=layer_flat,
        num_inputs=num_features,
        num_outputs=fc_size,
        use_relu=True)
    layer_fc1

    layer_fc2 = new_fc_layer(input=layer_fc1,
         num_inputs=fc_size,
         num_outputs=num_classes,
         use_relu=False)
    return layer_fc2

    
#weights = tf.Variable(tf.zeros([clip_size, num_classes]))
#biases = tf.Variable(tf.zeros([num_classes]))
#logits = tf.matmul(x, weights) + biases
logits = neural_network_model(x_image)

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
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
            o, c, cor, l, y_t = session.run([optimizer, cost, correct_prediction, y_pred_cls, y_true_cls], feed_dict=feed_dict_train)
            #print l
            #print y_t
            epoch_correct += np.sum(cor)
            epoch_total += cor.size
            epoch_loss += c
        print epoch_total,":epoch", i, "completed w/ loss", epoch_loss, "correct", epoch_correct, "and percentage" , 100.0 * float(epoch_correct) / float(epoch_total)

        test_index = randint(0, len(mat)-1000)
        print "test at", test_index
        test_clip, test_y_onehot, test_y = batch(mat, test_index, 10000)
        feed_dict_test = {x: test_clip, y_true: test_y_onehot, y_true_cls: test_y} 
        print_accuracy(feed_dict_test)


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

optimize(10)
#song = create(song, 100)
#print song, song.shape

session.close()
