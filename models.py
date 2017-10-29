import tensorflow as tf
import numpy as np
import os
import sys
import math

from audio import format_feedval, raw
from filter import savitzky_golay
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
        conv_offset = 20
        block = 3
        if self.onehot_mode == False:
            reg_image = tf.squeeze(reg_image, axis=[2])
            norm_image = tf.squeeze(norm_image, axis=[2])
        
        # hidden layers
        h = 10
        # hidden output layers
        d = 32
        n_residual_layers = 15

        num_blocks = 2
        num_layers = 14
        num_hidden = 128

        reg_channels = self.n_levels
        norm_channels = self.n_levels
        print("Regular Image", reg_image.shape, reg_channels)
        print("Normal Channels", norm_image.shape, norm_channels)

        image = tf.concat( [reg_image, norm_image] , axis=2)
        channels = norm_channels + reg_channels
        print("Image", image.shape, channels)
        
        '''    
        
        r1, rw1 = new_conv_layer(reg_image, reg_channels, block, reg_n_inputs - conv_offset, h)
        print("R1:", r1.shape, rw1.shape)

        n1, nw1 = new_conv_layer(norm_image, norm_channels, block, norm_n_inputs - conv_offset, h)
        print("N1:", n1.shape, nw1.shape)

        l1 = tf.concat( [r1, n1], axis=2)
        print("L1:", l1.shape)

        l2, w2 = new_conv_layer(l1, h, block, conv_offset, d)
        print("L2:", l2.shape)

        flat, features = flatten_layer(l2)
        print("Flat", flat.shape, features)

        fc_layers = nn_fc_layers(flat, features, n_target_classes, [512, 256])
        '''

        print(image)
        hl = image
        hs = []
        for b in range(num_blocks):
            for i in range(num_layers):
                #rate = 2**i
                rate = i+1
                name = 'b{}-l{}'.format(b, i)
                hl = dilated_conv1d(hl, num_hidden, rate=rate, name=name)
                hs.append(hl)
        
        outputs = conv1d(hl,
                         n_target_classes,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)
        print("Final Out:", outputs.shape)
        

        ''' 
        out_norm = []
        out_reg = []
        for i in range(reg_channels):
            input_norm = self.norm_list[i]
            input_reg = self.reg_list[i]

            #print("Norm", i,  input_norm.shape)
            #print("Reg", i, input_norm.shape)

            input_image = tf.concat( [input_norm, input_reg], axis=2)
            
            l1, w1 = new_conv_layer( input_image, 1 , 3, conv_offset , 1, 1, 2*h, False)
             
            print("L1", l1.shape)
            print("W1", w1.shape)
            
            r_layer = tf.nn.relu(l1)

            for _ in range(n_residual_layers):
                l_a, w_a = new_conv_layer( r_layer , 2*h, 1, 1, 1, 1, h, False)
                l_a = tf.nn.relu(l_a)
                #print("a", l_a.shape)
                l_b, w_b = new_conv_layer( l_a, h, 1, 1, 1, 1, h, False)
                l_b = tf.nn.relu(l_b)
                #print("b", l_b.shape)

                l_c, w_c = new_conv_layer( l_b, h, 1, 1, 1, 1, 2*h, False)
                l_c = tf.nn.relu(l_c)
                #print("c", l_c.shape)
                
                r_layer = tf.add(l_c, r_layer)
            l2, w2 = new_conv_layer( r_layer, 2*h, 1, 1, 1, 1, d, False)
            l2 = tf.nn.relu(l2)
            print("L2", l2.shape)
            
            flat, flat_features = flatten_layer(l2)
            #print("Flat", flat.shape, flat_features)

            if i == 0:
                out = flat
                features = flat_features
            else:
                out = tf.concat( [out, flat] , axis=1)
                features += flat_features
            #print("Out", out.shape, features)
        
        fc_layers = nn_fc_layers(out, features, n_target_classes, [256])
        #print("Fully Connected", fc_layers[-1].shape)
        '''
        
        '''
        reg_layers, reg_weights = nn_conv_layers(reg_image, reg_conv_sizes, conv_nodes, conv_pooling, n_channels, use_pooling)
        norm_layers, norm_weights = nn_conv_layers(norm_image, norm_conv_sizes, conv_nodes, conv_pooling, n_channels, use_pooling)

        reg_flat, reg_features = flatten_layer(reg_layers[-1])
        norm_flat, norm_features = flatten_layer(norm_layers[-1])
        
        flat = tf.concat( [reg_flat, norm_flat ] , axis=1)
        flat_features = reg_features + norm_features

        fc_layers = nn_fc_layers(flat, flat_features, n_target_classes, fc_nodes)
        #fc_layers = nn_fc_layers(norm_flat, norm_features, n_target_classes, fc_nodes)
        
        print(fc_layers)
        '''

        if (not os.path.isdir(self.save_dir)):
            print("  Normalized Mode", self.normalize_mode, "Onehot Mode", self.onehot_mode, "Multichannel Mode", self.multichannel_mode)

            print("  Image" , image.shape, "channels", channels)
            print("  Regular Image", reg_image.shape, "channels", reg_channels)
            print("  Normal Image", norm_image.shape, "channels", norm_channels)
            #print("  flat layer number of features", features)

            '''
            print("  Residual Layers", n_residual_layers, "Hidden Layers", h, "hidden output layers", d)
            print("  Layer Shape" , l3.shape)
            print("  Layer Weights" , w3.shape)
            for i, (r_s, n_s, f, p) in enumerate(zip(reg_conv_sizes, norm_conv_sizes, conv_nodes, conv_pooling)):
                print("  regular conv Layer", i, "filter:", r_s[0], r_s[1], "pooling:", p[0], p[1],
                        "number of channels", f, "use pooling", use_pooling)
                print("  normalized conv Layer", i, "filter:", n_s[0], n_s[1], "pooling:", p[0], p[1],"number of channels", 
                        f, "use pooling", use_pooling)
            
            print("  output layer", out.shape)
            '''
        #return fc_layers[-1]
        return outputs

   
    def __init__(self, level, receptive_field, data_location, n_levels ):
        self.batch_size = 512
        self.normalize_mode = False
        self.onehot_mode = False
        self.multichannel_mode = True
        self.use_pooling = False

        self.level = level
        self.receptive_field = receptive_field
        self.n_levels = n_levels
        
        self.in_bits = 8
        self.out_bits = 8

        #self.clip_size = 2*self.receptive_field+1
        self.clip_size = self.receptive_field
        self.middle_index = math.floor( float(self.receptive_field) / 2.0)
        #print("Middle Index", self.middle_index)
        self.n_input_classes = 2**(self.in_bits)
        self.input_classes_max = self.n_input_classes - 1
        self.n_target_classes = 2**(self.out_bits)
        self.target_classes_max = self.n_target_classes - 1

        self.name = "model_" + str(level) + "_r" + str(self.receptive_field)
        self.save_dir = data_location + "/" + self.name
        self.save_file = self.save_dir + "/" + self.name + ".ckpt"

        input_level = tf.placeholder(tf.int64, [None,self.clip_size])
        input_all = tf.placeholder(tf.int64, [self.n_levels, None,self.clip_size])
        target_class = tf.placeholder(tf.int64, [None])

        self.input_level = input_level
        self.input_all = input_all
        self.target_class = target_class

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

        logits = self.neural_network_model(regular_inputs, reg_n_inputs, normalized_inputs, norm_n_inputs, nn_n_targets, n_channels, self.use_pooling)

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
    '''
    def test_io_onehot(self, x, ytrue_class, x_list):
        np.set_printoptions(threshold=np.inf)
        x = np.reshape(x, (-1, self.clip_size))
        ytrue_class = np.reshape(ytrue_class, (-1))
        print("Testing:",  self.name, x.shape, ytrue_class.shape)
        tests = 1
        for i in range(int(len(x)/2), int(len(x)/2 + 1), tests):
            feed_dict_test = {self.input_level: x[i:i+tests,:],
                              self.target_class: ytrue_class[i:i+tests],
                              self.input_all: x_list[i:i+tests,:,:] }
            inp, mid, norm, norm_pos, norm_one, one, targ_c, in_scale, tar_nor, tar_nor_pos, tar_nor_pos_one, tar, image, one_image = self.sess.run(
                    [self.input_level, self.middle, self.normalized, self.normalized_pos, 
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
    '''

    def train(self, x_list, ytrue_class, epochs=1 ):
        #x = np.reshape(x, (-1, self.clip_size))
        ytrue_class = np.reshape(ytrue_class, (-1))
        print("Trainging:",  self.name, x_list.shape, ytrue_class.shape, "epochs:", epochs)
        
        #for e in range(epochs):
        e = 0
        while ((e < epochs) and (self.best_accuracy < 100.1 )):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            for i in range(0, len(ytrue_class), self.batch_size):
                feed_dict_train = {self.target_class: ytrue_class[i:i+self.batch_size],
                                   self.input_all: x_list[:,i:i+self.batch_size,:] }
                
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

    def generate(self, song, index_list, frequency_list, sample_length):
        field_size = abs(np.amin(index_list))
        
        print("Sample size:", sample_length, "Field Size", field_size)
        
        #y_generated = np.append(song, np.zeros(sample_length))
        #x_size = song.size
        y_generated = np.append(np.zeros(field_size), np.zeros(sample_length))
        x_size = field_size
        
        print("Generating with seed:", song.shape, x_size)
        print("Index list:", index_list.shape)
        y_size = y_generated.size
        if(y_size <= field_size):
            raise ValueError('Sample Length too small for receptive field')
            sys.exit()
        print("Y generate", y_generated.shape, y_size)
        #feed_val = np.empty( (self.n_levels, 1 , self.receptive_field) )
        #index = np.reshape(index_list, (self.n_levels, 1, self.receptive_field))
        #print(index.shape, index)
        for i in range(x_size, y_size):
            #print( y_generated[i-field_size:i+1])
            #y_generated[i-field_size:i+1] = savitzky_golay(y_generated[i-field_size:i+1], 41, 5) 
            feed_val = format_feedval(y_generated[i-field_size:i+1], frequency_list, index_list,
                    1, self.n_levels)
            #print( y_generated[i-field_size:i+1])
            #print()
            #print("Feed val", feed_val.shape)
            feed_dict_gen = { self.input_all: feed_val }
            y_g = self.sess.run( [self.prediction_value], feed_dict=feed_dict_gen)
            y_generated[i] = raw(y_g[0])
            #print("y[", i, "] = ", y_g, y_generated[i])
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
