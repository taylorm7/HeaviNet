import tensorflow as tf
import numpy as np
import os
import sys
import math

def nn_layer(input_layer, n_nodes_in, n_nodes, output_layer=False):
    hl_weight = tf.Variable(tf.random_normal([n_nodes_in, n_nodes]))
    hl_bias = tf.Variable(tf.random_normal([n_nodes]))

    layer =  tf.matmul(input_layer, hl_weight) + hl_bias

    if(not output_layer):
        layer = tf.nn.relu(layer)
    return layer



# Weights, bias, and convolutional layer helper methods referenced and modified from:
# https://github.com/Hvass-Labs/TensorFlow-Tutorials -> 02_Convolutional_Neural_Network.ipynb
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_width,
                   filter_height, # Width and height of each filter.
                   pool_width,
                   pool_height,
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_width, filter_height, num_input_channels, num_filters]

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
                         padding='VALID')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    if pool_height <= 1 and pool_width <= 1:         
        use_pooling = False

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, pool_width, pool_height, 1],
                               strides=[1, pool_width, pool_height, 1],
                               padding='VALID')

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

def nn_fc_layers(input, n_input_classes, n_target_classes, n_nodes):
    layers = []
    for i, node in enumerate( n_nodes ):
        if i == 0:
            layers.append(new_fc_layer(input, n_input_classes, node, use_relu=True))
        else:
            layers.append(new_fc_layer(layers[i-1], n_nodes[i-1], node, use_relu=True))

    layers.append(new_fc_layer(layers[-1], n_nodes[-1], n_target_classes, use_relu=False))
    return layers

def nn_conv_layers(data_image, filter_sizes, filter_nodes, pooling_sizes, use_pooling ):
    layers = []
    weights = []
    
    for i, (size, num, pool) in enumerate( zip(filter_sizes, filter_nodes, pooling_sizes) ):
        if i == 0:
            l, w = new_conv_layer(data_image, 1, size[0], size[1], pool[0], pool[1], num, use_pooling)
        else:
            l, w = new_conv_layer(layers[i-1], filter_nodes[i-1], size[0], size[1], pool[0], pool[1], num, use_pooling)
        layers.append(l)
        weights.append(w)
    return layers, weights

class Model(object):
    def perceptron_nn(self, input, n_input_classes, n_target_classes, clip_size, n_nodes):
        layers = []
        for i, nodes in enumerate( n_nodes ):
            if i == 0:
                layers.append(nn_layer(input, n_input_classes*clip_size, n_nodes[0]))
            else:
                layers.append(nn_layer(layers[i-1], n_nodes[i-1], n_nodes[i]))
        output_layer = nn_layer(layers[-1], n_nodes[-1], n_target_classes, output_layer=True)
        return output_layer

    def neural_network_model(self, data_image, n_input_classes, n_target_classes, use_pooling):
        '''
        conv_nodes = [ 1 , 4 ]
                    # (clip_size, n_input_classes)
        conv_sizes =   [ (self.clip_size , (self.level+1) ), 
                         (1 ,  math.floor( n_input_classes / (self.level+1))) ]

        conv_pooling = [ ( 1 , 1 ), 
                         ( 1 , 1 ) ]
        
        fc_nodes =   [ 2**(self.level+3) , 2**(self.level+2) ]
        '''
        
        conv_nodes = [ 8 , 16 ]
        # input is formated in tensor: (clip_size, n_input_classes)
        conv_sizes =   [ (self.clip_size , self.level + 1),
                         ( 1, (self.level+1)*2 ) ]
        conv_pooling = [ ( 1 , 1 ), (1, 1) ]
        fc_nodes =   [ 2**(self.level+3) , 2**(self.level+2) ]

        
        conv_layers, conv_weights = nn_conv_layers(data_image, conv_sizes, conv_nodes, conv_pooling, use_pooling)
        conv_flat, n_features = flatten_layer(conv_layers[-1])
        fc_layers = nn_fc_layers(conv_flat, n_features, n_target_classes, fc_nodes)
    
        if (not os.path.isdir(self.save_dir)):
            print("  Normalized Mode", self.normalize_mode)
            for i, (s, f, p) in enumerate(zip(conv_sizes, conv_nodes, conv_pooling)):
                print("  conv Layer", i, "filter:", s[0], s[1], "pooling:", p[0], p[1],
                        "number of channels", f, "use pooling", use_pooling)
            print("  flat layer number of features", n_features)
            for i, n in enumerate(fc_nodes):
                print("  fully connected layer", i, "number of nodes", n)
            print("  targets", n_target_classes)

        return fc_layers[-1]


    def __init__(self, level, receptive_field, data_location, 
                 batch_size=128, normalize_mode=True, use_pooling=False ):

        clip_size = 2*receptive_field+1
        n_input_classes = 2**(level+1)
        input_classes_max = n_input_classes - 1
        n_target_classes = 2**(level+2)
        target_classes_max = n_target_classes - 1

        self.level = level
        self.batch_size = batch_size
        self.receptive_field = receptive_field
        self.clip_size = clip_size

        self.name = "model_" + str(level) + "_r" + str(receptive_field)
        self.save_dir = data_location + "/" + self.name
        self.save_file = self.save_dir + "/" + self.name + ".ckpt"

        inputs = tf.placeholder(tf.int64, [None,clip_size])
        target_class = tf.placeholder(tf.int64, [None])

        #create onehot value for non-normalized inputs
        onehot = tf.one_hot(inputs, n_input_classes)
        onehot_image = tf.reshape(
                onehot, [-1, clip_size, n_input_classes,  1])
        onehot = tf.reshape(onehot, (-1, clip_size*n_input_classes))
        # create regular onehot values for target
        target = tf.one_hot(target_class, n_target_classes)
        target = tf.reshape(target, (-1, n_target_classes))

        #normalized inputs and target, along with corresponding onehot

        # slices tensor from middle value -> [0, middle_index] 
        # to end of None -> [-1(end), 1 (one value only)] 
        middle = tf.slice(inputs, [0,  receptive_field] , [-1, 1])
        middle_ = tf.reshape(middle, [-1] )
        normalized = tf.subtract(inputs, middle)
        normalized_pos = normalized + input_classes_max
         
        input_norm_onehot_range = input_classes_max*2+1
        normalized_onehot = tf.one_hot(normalized_pos, input_norm_onehot_range)
        
        normalized_onehot_image = tf.reshape(
                normalized_onehot, [-1, clip_size, input_norm_onehot_range,  1]) 
        normalized_onehot = tf.reshape(normalized_onehot, (-1, clip_size*input_norm_onehot_range))
        normalized_onehot = tf.cast(normalized_onehot, tf.float32)
                
        # normalize targets based on difference from input values to scaled targets
        target_norm_onehot_range = target_classes_max*2+1
        inputs_scaled = tf.multiply(middle_ , 2)
        inputs_scaled_ = tf.reshape(inputs_scaled, [-1])
        
        target_normalized = tf.subtract(target_class, inputs_scaled)
        target_normalized_pos = target_normalized + target_classes_max
        target_normalize_pos_ = tf.reshape(target_normalized_pos, [-1])
        target_normalized_onehot = tf.one_hot(
                target_normalized_pos, target_norm_onehot_range)
        target_normalized_onehot = tf.reshape(
                target_normalized_onehot, (-1, target_norm_onehot_range) )

        #create list of perceptron levels with the number of corresponding nodes per level
        #n_nodes = [ (level+1)*400, (level+1)*100, (level+1)*50, ]
        #n_nodes = [ 1024, 512, 256, ]


        if normalize_mode == True:
            nn_inputs = normalized_onehot_image
            nn_targets = target_normalized_onehot
            nn_target_class = target_normalize_pos_
            nn_n_inputs = input_norm_onehot_range
            nn_n_targets = target_norm_onehot_range
            self.normalize_mode = True
        else:
            nn_inputs = onehot_image
            nn_targets = target
            nn_target_class = target_class
            nn_n_inputs = n_input_classes
            nn_n_targets = n_target_classes
            self.normalize_mode = False

        
        #logits = self.perceptron_nn(nn_inputs, nn_n_inputs, nn_n_targets, clip_size, n_nodes)
        logits = self.neural_network_model(nn_inputs, nn_n_inputs, nn_n_targets, use_pooling)

        prediction = tf.nn.softmax(logits)
        prediction_class = tf.argmax(prediction, dimension=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=nn_targets)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        correct_prediction = tf.equal(nn_target_class, prediction_class)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32) )

        if normalize_mode == True:
            prediction_value = (prediction_class - target_classes_max) + inputs_scaled_
        else:
            prediction_value = prediction_class
        
        sess = tf.Session()
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        self.clip_size = clip_size

        self.inputs = inputs
        self.target = target

        self.target_class = target_class
        self.onehot = onehot
        self.onehot_image = onehot_image
        
        self.middle = middle
        self.normalized = normalized
        self.normalized_pos = normalized_pos
        self.normalized_onehot = normalized_onehot
        self.normalized_onehot_image = normalized_onehot_image

        self.inputs_scaled = inputs_scaled
        self.target_normalized = target_normalized
        self.target_normalized_pos = target_normalized_pos
        self.target_normalized_onehot = target_normalized_onehot

        self.optimizer = optimizer
        self.cost = cost
        self.accuracy = accuracy
        self.correct_prediction = correct_prediction
        self.best_accuracy = 0
        self.loss_change = 0

        self.prediction_value = prediction_value

        self.sess = sess
        self.saver = saver
        

        if( os.path.isdir(self.save_dir) ):
            print("Loading previous:", self.name)
            self.load()
        else:
            os.makedirs( self.save_dir )
            print("Creating level directory at:", self.save_dir)

    def test_io_onehot(self, x, ytrue_class):
        np.set_printoptions(threshold=np.inf)
        x = np.reshape(x, (-1, self.clip_size))
        ytrue_class = np.reshape(ytrue_class, (-1))
        print("Testing:",  self.name, x.shape, ytrue_class.shape)
        for i in range(int(len(x)/2), int(len(x)/2 + 1), 10):
            feed_dict_test = {self.inputs: x[i:i+10,:],
                               self.target_class: ytrue_class[i:i+10] }
            inp, mid, norm, norm_pos, norm_one, one, targ_c, in_scale, tar_nor, tar_nor_pos, tar_nor_pos_one, tar, image, one_image = self.sess.run(
                    [self.inputs, self.middle, self.normalized, self.normalized_pos, 
                     self.normalized_onehot, self.onehot, self.target_class, self.inputs_scaled,
                     self.target_normalized, self.target_normalized_pos, 
                     self.target_normalized_onehot, self.target, self.normalized_onehot_image,
                     self.onehot_image],
                    feed_dict=feed_dict_test)
            print("inputs\n", inp)
            print("middle value of receptive_field\n", mid)
            print("normalized inputs\n", norm)
            print("positive normalized\n", norm_pos)
            print("onehot normalized inputs\n", norm_one)
            print("onehot normalized image\n", image)
            print("regular onehot image\n", one_image)
            print("regular onehot inputs\n", one)
            
            print("inputs scaled to next level\n", in_scale)
            print("target classes\n", targ_c)
            print("targets normalized\n", tar_nor)
            print("positive normalized targets\n", tar_nor_pos)
            print("onehot normalized targets\n", tar_nor_pos_one)
            print("regular onehot targets\n", tar)


    def train(self, x, ytrue_class, epochs=1 ):
        x = np.reshape(x, (-1, self.clip_size))
        ytrue_class = np.reshape(ytrue_class, (-1))
        print("Trainging:",  self.name, x.shape, ytrue_class.shape, "epochs:", epochs)
        
        #for e in range(epochs):
        e = 0
        while ((e < epochs) and (self.best_accuracy < 100 )):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            for i in range(0, len(x), self.batch_size):
                feed_dict_train = {self.inputs: x[i:i+self.batch_size,:],
                                   self.target_class: ytrue_class[i:i+self.batch_size] }
                
                # train without calculating accuracy
                #_, c = self.sess.run([self.optimizer, self.cost],
                #                        feed_dict = feed_dict_train)
                
                # train while calculating epoch accuracy
                _, c, correct = self.sess.run([self.optimizer, self.cost, self.correct_prediction ],
                                        feed_dict = feed_dict_train)
                epoch_correct+= np.sum(correct)
                epoch_total+= correct.size
                
                epoch_loss+= c

            print(" epoch", e, "loss", epoch_loss)
            e = e+1

            if (epoch_total != 0):
                epoch_accuracy = 100.0 * float(epoch_correct) / float(epoch_total)
                print(" accuracy:", epoch_accuracy)
                if epoch_accuracy > self.best_accuracy :
                    self.best_accuracy = epoch_accuracy

    def generate(self, x_seed):
        x_seed = np.reshape(x_seed, (-1, self.clip_size))
        print("Generating with seed:", x_seed.shape)
        y_generated = []
        for i in range(0, len(x_seed), self.batch_size):
            feed_dict_gen = {self.inputs: x_seed[i:i+self.batch_size,:]}
            y_g = self.sess.run( [self.prediction_value], feed_dict=feed_dict_gen)
            y_generated = np.append(y_generated, y_g)
        print("Generated song:",  len(y_generated))
        return y_generated

    def save(self, close=False):
        self.saver.save(self.sess, self.save_file)
        print("Saving:", self.name)
        if close==True:
            self.sess.close()

    def load(self):
        if os.path.isdir(self.save_dir):
            try:
                self.saver.restore(self.sess, self.save_file)
            except:
                print("Failed to load previous session")
                os.rmdir(self.save_dir) 
                sys.exit()
            return True
        else:
            print("Failed loading:", self.name)
            return False

    def close(self):
        print("Closing" , self.name)
        self.sess.close()
