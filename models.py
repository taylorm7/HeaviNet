import tensorflow as tf
import numpy as np
import os
import sys
import math

from audio import format_feedval, raw
#from filter import savitzky_golay
from layers import conv1d, dilated_conv1d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
            #tf.Variable(tf.truncated_normal(shape=[length], stddev=0.44))


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_width,
                   filter_height, # Width and height of each filter.
                   num_filters,        # Number of filters.
                   pool_width=0,
                   pool_height=0,
                   use_pooling=False):  # Use 2x2 max-pooling.

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
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    
    #layer = tf.nn.relu(layer)

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

def nn_conv_layers(data_image, filter_sizes, filter_nodes, pooling_sizes, num_input_channels, use_pooling ):
    layers = []
    weights = []
    
    for i, (size, num, pool) in enumerate( zip(filter_sizes, filter_nodes, pooling_sizes) ):
        if i == 0:
            l, w = new_conv_layer(data_image, num_input_channels, size[0], size[1], num, pool[0], pool[1], use_pooling)
        else:
            l, w = new_conv_layer(layers[i-1], filter_nodes[i-1], size[0], size[1], num, pool[0], pool[1], use_pooling)
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

    def format_in_flat(self, in_level):
        image = tf.reshape( in_level, [-1, self.clip_size, 1, 1] ) 
        image = tf.cast(image, tf.float32)
        return image, 1
        
    def format_in_onehot(self, in_level):
        onehot = tf.one_hot(in_level, self.n_input_classes)
        onehot_image = tf.reshape( onehot, [-1, self.clip_size, self.n_input_classes,  1])
        return onehot_image, self.n_input_classes

    def format_target(self, in_level, target_c):
        target = tf.one_hot(target_c, self.n_target_classes)
        target = tf.reshape(target, (-1, self.n_target_classes))
        return target, target_c, self.n_target_classes

    def format_in_norm_flat(self, in_level):
        middle = tf.slice(in_level, [0,  self.middle_index] , [-1, 1])
        middle_ = tf.reshape(middle, [-1] )
        normalized = tf.subtract(in_level, middle)
        normalized_pos = normalized + self.input_classes_max

        normalized_image = tf.reshape(normalized_pos, [-1, self.clip_size, 1, 1] )
        normalized_image = tf.cast(normalized_image, tf.float32)
        return normalized_image, 1

    def format_in_norm_onehot(self, in_level):
        middle = tf.slice(in_level, [0,  self.middle_index] , [-1, 1])
        middle_ = tf.reshape(middle, [-1] )
        normalized = tf.subtract(in_level, middle)
        normalized_pos = normalized + self.input_classes_max
        
        input_norm_onehot_range = self.input_classes_max*2+1
        normalized_onehot = tf.one_hot(normalized_pos, input_norm_onehot_range)
        
        normalized_onehot_image = tf.reshape( normalized_onehot, [-1, self.clip_size, input_norm_onehot_range,  1])
        return normalized_onehot_image, input_norm_onehot_range

    def format_target_norm(self, in_level, target_c):
        middle = tf.slice(in_level, [0,  self.middle_index] , [-1, 1])
        middle_class = tf.reshape(middle, [-1] )
        
        target_norm_onehot_range = self.target_classes_max*2+1
        inputs_scaled = tf.multiply(middle_class , 2)
        inputs_scaled_class = tf.reshape(inputs_scaled, [-1])
        
        target_normalized = tf.subtract(target_c, inputs_scaled)
        target_normalized_pos = target_normalized + self.target_classes_max
        target_normalized_class = tf.reshape(target_normalized_pos, [-1])
        target_normalized_onehot = tf.one_hot( target_normalized_pos, target_norm_onehot_range)
        target_normalized_onehot = tf.reshape( target_normalized_onehot, (-1, target_norm_onehot_range) )
        
        self.inputs_scaled_class = inputs_scaled_class
        return target_normalized_onehot, target_normalized_class, target_norm_onehot_range
 
    def neural_network_model(self, reg_image, reg_n_inputs, norm_image, norm_n_inputs, n_target_classes, n_channels, use_pooling):
        if self.onehot_mode == False:
            reg_image = tf.squeeze(reg_image, axis=[2])
            norm_image = tf.squeeze(norm_image, axis=[2])
        
        # hidden layers
        h = 10
        # hidden output layers
        d = 32
        n_residual_layers = 15

        num_blocks = 1
        num_layers = int(self.n_levels/2)
        num_hidden = 32

        reg_channels = self.n_levels
        norm_channels = self.n_levels
        print("Blocks", num_blocks, "Layers", num_layers, "Hidden", num_hidden);

        print("Regular Image", reg_image.shape, reg_channels)
        print("Normal Channels", norm_image.shape, norm_channels)

        image = tf.concat( [reg_image, norm_image ] , axis=2)
        channels = norm_channels + reg_channels
        print("Image", image.shape, channels)
        
        hl = tf.transpose(image, perm=[0, 2, 1] )
        print("Input image", hl.shape)
        hs = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                #rate = 1
                name = 'b{}-l{}'.format(b, i)
                hl = dilated_conv1d(hl, num_hidden, rate=rate, name=name)
                hs.append(hl)
        
        outputs = dilated_conv1d(hl, n_target_classes, rate=1, name='out')
        print("Out:", outputs.shape)
        outputs = tf.transpose(outputs, perm=[0, 2, 1] )
        print("Out Transpose:", outputs.shape)
        outputs = conv1d(outputs,
                         1,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)
        outputs = tf.squeeze(outputs, axis=[2])
        print("Final Out:", outputs.shape)

        #flat, features = flatten_layer(l2)
        #print("Flat", flat.shape, features)

        #fc_layers = nn_fc_layers(flat, features, n_target_classes, [512, 256])
 

        if (not os.path.isdir(self.save_dir)):
            print("  Normalized Mode", self.normalize_mode, "Onehot Mode", self.onehot_mode, "Multichannel Mode", self.multichannel_mode)

            print("  Image" , image.shape, "channels", channels)
            print("  Regular Image", reg_image.shape, "channels", reg_channels)
            print("  Normal Image", norm_image.shape, "channels", norm_channels)
            #print("  flat layer number of features", features)

        #return fc_layers[-1]
        return outputs

    def wavenet_model(self, image):
        num_blocks = 2
        num_layers = 14
        num_hidden = 128
    
        image = tf.reshape(image, (-1, self.batch_size, 1))
        image = tf.cast(image, tf.float32)
        print("Image", image.shape, image.dtype)

        h = image
        hs = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                h = dilated_conv1d(h, num_hidden, rate=rate, name=name)
                hs.append(h)

        outputs = conv1d(h,
                         self.n_target_classes,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)
        
        outputs = tf.reshape(outputs, (-1, self.n_target_classes))

        if (not os.path.isdir(self.save_dir)):
            print("  Wavenet Mode")
            print("  Normalized Mode", self.normalize_mode, "Onehot Mode", self.onehot_mode, "Multichannel Mode", self.multichannel_mode)
            print("  Image" , image.shape)
            print("  Outputs" , outputs.shape)
            print("  Batch size" , self.batch_size, "Batch Hot", self.batch_hot, "Start", self.batch_start, "Stop", self.batch_stop)
        return outputs
   
    def __init__(self, level, receptive_field, data_location, n_levels ):
        self.batch_size = 1000
        self.batch_hot = 500
        self.normalize_mode = False
        self.onehot_mode = False
        self.multichannel_mode = True
        self.use_pooling = False
        self.wavenet_test = True

        self.level = level
        self.receptive_field = receptive_field
        self.n_levels = n_levels
        self.n_epochs = tf.get_variable("n_epochs", shape=[], dtype=tf.int32, initializer = tf.zeros_initializer)
        
        self.in_bits = 8
        self.out_bits = 8

        self.batch_iterate = round((self.batch_size - self.batch_hot ) / 2)
        self.batch_start = self.batch_iterate
        self.batch_stop = self.batch_iterate + self.batch_hot

        self.clip_size = 2*self.receptive_field+1
        #self.clip_size = self.receptive_field
        self.middle_index = math.floor( float(self.receptive_field) / 2.0)
        #print("Middle Index", self.middle_index)
        self.n_input_classes = 2**(self.in_bits)
        self.input_classes_max = self.n_input_classes - 1
        self.n_target_classes = 2**(self.out_bits)
        self.target_classes_max = self.n_target_classes - 1

        if self.wavenet_test == False:
            self.model_string = "h"+str(self.batch_size)+"_"
        else:
            self.model_string = "w"+str(self.batch_size)+"_"
        self.name = self.model_string + str(level) + "_r" + str(self.receptive_field)
        self.seed_name = self.model_string + str(level-1) + "_r" + str(self.receptive_field)
        self.save_dir = data_location + "/" + self.name
        self.save_file = self.save_dir + "/" + self.name + ".ckpt"

        input_level = tf.placeholder(tf.float64, [None,self.clip_size])
        input_all = tf.placeholder(tf.float64, [self.n_levels, None,self.clip_size])
        target_class = tf.placeholder(tf.int64, [None])
        input_class = tf.placeholder(tf.float64, [None])

        self.input_level = input_level
        self.input_all = input_all
        self.target_class = target_class
        self.input_class = input_class

        if self.onehot_mode == False:
            normalized_call = self.format_in_norm_flat
            regular_call = self.format_in_flat
        else:
            normalized_call = self.format_in_norm_onehot
            regular_call = self.format_in_onehot

        if self.normalize_mode == False:
            out_call = self.format_target
        elif self.normalize_mode == True:
            out_call = self.format_target_norm

        regular_inputs = []
        regular_level = []
        normalized_inputs = []
        normalized_level = []

        for i in range(self.n_levels):
            #if i != self.level:
            regular_level, _ = regular_call(input_all[i])
            regular_inputs.append(regular_level)
            normalized_level, _ = normalized_call(input_all[i])
            normalized_inputs.append(normalized_level)

        #if self.level == 0:
        #    regular_inputs = normalized_inputs
        #else:
        #    del regular_inputs[self.level]
        
        self.reg_list = regular_inputs
        self.norm_list = normalized_inputs

        regular_inputs = tf.concat( regular_inputs, axis=3)
        _, reg_n_inputs = regular_call(input_level)
        normalized_inputs = tf.concat( normalized_inputs, axis=3)
        _, norm_n_inputs = normalized_call(input_level)
        nn_targets, nn_target_class, nn_n_targets = out_call(input_level, target_class)
        n_channels = self.level
        
        #if self.level ==0:
        #    reg_n_inputs = norm_n_inputs
        
        if self.wavenet_test == False:
            logits = self.neural_network_model(regular_inputs, reg_n_inputs, normalized_inputs, norm_n_inputs, nn_n_targets, n_channels, self.use_pooling)
        else:
            self.logits_original = self.wavenet_model(input_class)
            
            self.in_backwards = tf.reverse(input_class,[0])
            with tf.variable_scope('backwards'):
                self.logits_backwards = self.wavenet_model( self.in_backwards)
            self.logits_b = tf.reverse(self.logits_backwards, [0])
            self.logits = tf.reduce_sum( tf.stack( [self.logits_original, self.logits_b], axis=0), axis=0)
            #self.logits = self.logits[self.batch_start:self.batch_stop, : ]
            #nn_targets = nn_targets[self.batch_start:self.batch_stop, : ]
            logits = self.logits

        prediction = tf.nn.softmax(logits)
        prediction_class = tf.argmax(prediction, dimension=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=nn_targets)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
        correct_prediction = tf.equal(nn_target_class, prediction_class)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32) )

        if self.normalize_mode == True:
            prediction_value = (prediction_class - self.target_classes_max) + self.inputs_scaled_class
        else:
            prediction_value = prediction_class
        
        sess = tf.Session()
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

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

    def train(self, x_list, ytrue_class, x, epochs=1 ):
        #x = np.reshape(x, (-1, self.clip_size))
        ytrue_class = np.reshape(ytrue_class, (-1))
        print("Trainging:",  self.name, x_list.shape, ytrue_class.shape, "epochs:", epochs)
        
        #for e in range(epochs):
        e = 0
        print("Previous Epochs", self.n_epochs.eval(session=self.sess) )
        inc_epochs = self.n_epochs.assign(self.n_epochs + epochs)
        inc_epochs.op.run(session=self.sess)
        while ((e < epochs) and (self.best_accuracy < 100.1 )):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            for i in range(0, len(ytrue_class), self.batch_iterate):
                if i + self.batch_size >= len(ytrue_class):
                    continue
                feed_dict_train = {
                                   #self.target_class: ytrue_class[i+self.batch_start:i+self.batch_stop],
                                   self.target_class: ytrue_class[i:i+self.batch_size],
                                   self.input_all: x_list[:,i:i+self.batch_size,:] ,
                                   self.input_class: x[i:i+self.batch_size]
                                   }
                # test logits value
                '''
                _, c, inp, in_b,l_back, l_b, l_o, l  = self.sess.run([self.optimizer,self.cost, 
                    self.input_class, 
                    self.in_backwards, 
                    self.logits_backwards, 
                    self.logits_b,
                    self.logits_original,
                    self.logits
                    ],
                                        feed_dict = feed_dict_train)
                print("Initial", inp.shape, inp  ) 
                print("Input Backwards", in_b.shape, in_b )
                print("Logits Forwards/backwards", l_back.shape, l_back)
                print("Logits Backwards", l_b.shape, l_b  )
                print("Logits Original", l_o.shape, l_o )
                print("Logits", l.shape, l)
                sys.exit()
                '''
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
 

    def generate(self, song_list, index_list, frequency_list, seed):
        y_generated = np.zeros(len(song_list[0]))
        print("Generating with seed_list:", song_list.shape)
        print("seed:", seed.shape, seed)
        print("Index list:", index_list.shape)
        print("Y generate", y_generated.shape )
        for i in range(0, len(seed), self.batch_hot):
            if i + self.batch_size >= len(seed):
                continue
            #feed_dict_gen = { self.input_all: song_list[:,i:i+self.batch_size,:] }
            feed_dict_gen = {self.input_class: seed[i:i+self.batch_size],
                                   self.input_all: song_list[:,i:i+self.batch_size,:] }
            y_g = self.sess.run( [self.prediction_value], feed_dict=feed_dict_gen)
            if i == 0:
                #print("First batch", i, i+self.batch_stop)
                #print(y_g[0][i:i+self.batch_stop])
                y_generated[i:i+self.batch_stop] = raw(y_g[0][i:i+self.batch_stop])
            else:
                #print(i, i+self.batch_start, i + self.batch_stop)
                #print(y_g[0][self.batch_start:self.batch_stop])
                y_generated[i+self.batch_start:i+self.batch_stop] = raw( y_g[0][self.batch_start:self.batch_stop] )
            #print("y[", i, "] = ", y_g[0], y_generated[i:i+self.batch_size])
        prev_epochs = self.n_epochs.eval(session=self.sess)
        print("Generated song:",  len(y_generated), "with Epochs", prev_epochs)
        return y_generated, prev_epochs

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
