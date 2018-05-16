import tensorflow as tf
import numpy as np
import os
import sys
import math

from audio import format_feedval, raw
from layers import conv1d, dilated_conv1d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model: neural network model for HeaviNet level
# inputs: init function
# level, integer of corresponding level
# receptive_field, integer of receptive field
# data_location, string value of data location to store neural network
# n_levels, integer value of number of total levels
# outputs:
# neural network stored in data_location with save command and loaded with load command
class Model(object):
    # format_target: make target matrix into onehot format
    # inputs
    # target_c, matrix of target classes to be formatted into onehot
    # outputs:
    # target, matrix of targets with a width of n_target_classes
    # target_c, matrix of origianl target classes
    # n_target_classes, integer of the number of targets (256 for 8 bit targets)
    def format_target(self, target_c):
        target = tf.one_hot(target_c, self.n_target_classes)
        target = tf.reshape(target, (-1, self.n_target_classes))
        return target, target_c, self.n_target_classes
    
    # wavenet_model: creates network structure according to fast-wavenet implementation
    # https://github.com/tomlepaine/fast-wavenet
    # inputs
    # image, matrix of a given batch of inputs with dimenstions [batch_size]
    # outputs:
    # outputs, matrix of computed results with shape [batch_size, n_target_classes]
    def wavenet_model(self, image):
        num_blocks = 2
        num_layers = 14
        num_hidden = 128
        
        # reshape image according to fast-wavenet model
        image = tf.reshape(image, (-1, self.batch_size, 1))
        image = tf.cast(image, tf.float32)
        print("Image", image.shape, image.dtype)

        # compute layers according to num_layers and num_blocks
        h = image
        hs = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                h = dilated_conv1d(h, num_hidden, rate=rate, name=name)
                hs.append(h)
        # compute final layer acording to n_target_classes
        outputs = conv1d(h,
                         self.n_target_classes,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)
        
        # reshape output for softmax computation
        outputs = tf.reshape(outputs, (-1, self.n_target_classes))

        if (not os.path.isdir(self.save_dir)):
            print("  Wavenet Model")
            print("  Image" , image.shape)
            print("  Outputs" , outputs.shape)
            print("  Batch size" , self.batch_size )
        return outputs

    # init: neural network model for HeaviNet level
    # inputs: init function
    # level, integer of corresponding level
    # receptive_field, integer of receptive field
    # data_location, string value of data location to store neural network
    # n_levels, integer value of number of total levels
    # outputs:
    # neural network stored in data_location with save command and loaded with load command  
    def __init__(self, level, receptive_field, data_location, n_levels ):
        self.batch_size = 1000

        self.level = level
        self.receptive_field = receptive_field
        self.n_levels = n_levels
        # store n_epochs with neural network
        self.n_epochs = tf.get_variable("n_epochs", shape=[], dtype=tf.int32, initializer = tf.zeros_initializer)
        
        self.in_bits = 8
        self.out_bits = 8

        self.n_target_classes = 2**(self.out_bits)
        self.target_classes_max = self.n_target_classes - 1
        
        # format name to save multiple levels in single directory
        self.model_string = "w"+str(self.batch_size)+"_"
        self.name = self.model_string + str(level) + "_r" + str(self.receptive_field)
        self.seed_name = self.model_string + str(level-1) + "_r" + str(self.receptive_field)
        self.save_dir = data_location + "/" + self.name
        self.save_file = self.save_dir + "/" + self.name + ".ckpt"

        # inputs and targets for training or generation
        target_class = tf.placeholder(tf.int64, [None])
        input_class = tf.placeholder(tf.float64, [None])

        self.target_class = target_class
        self.input_class = input_class
        
        # format targets into class and onehot formats
        nn_targets, nn_target_class, nn_n_targets = self.format_target(target_class)

        # compute original wavenet logits

        self.logits = self.wavenet_model(input_class)

        # compute reversed wavenet logits
        #self.in_backwards = tf.reverse(input_class,[0])
        #with tf.variable_scope('backwards'):
        #    self.logits_backwards = self.wavenet_model( self.in_backwards)
        #self.logits_b = tf.reverse(self.logits_backwards, [0])

        # sum logits and backwards logits
        #self.logits = tf.reduce_sum( tf.stack( [self.logits_original, self.logits_b], axis=0), axis=0)
        logits = self.logits

        # compute prediction and prediction class based on logits
        prediction = tf.nn.softmax(logits)
        prediction_class = tf.argmax(prediction, dimension=1)

        # compute cross entropy and correpsonding optimization
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=nn_targets)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
        correct_prediction = tf.equal(nn_target_class, prediction_class)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32) )

        self.prediction_value = prediction_class
        
        sess = tf.Session()
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        self.optimizer = optimizer
        self.cost = cost
        self.accuracy = accuracy
        self.correct_prediction = correct_prediction
        self.best_accuracy = 0
        self.loss_change = 0

        self.sess = sess
        self.saver = saver
        

        if( os.path.isdir(self.save_dir) ):
            print("Loading previous:", self.name)
            self.load()
        else:
            os.makedirs( self.save_dir )
            print("Creating level directory at:", self.save_dir)

    # train: train model on inputs according to number of epochs
    # inputs: 
    # x_list, list of matricies containing all input levels
    # ytrue_class, matrix of target values for corresponding x matrix input values
    # x, matrix of level inputs
    # epochs, integer of number of epochs to train
    # outputs:
    # none, trains neural network over inputs and targets
    def train(self, ytrue_class, x, epochs=1 ):
        ytrue_class = np.reshape(ytrue_class, (-1))
        print("Trainging:",  self.name, "Inputs", x.shape, "Targets", ytrue_class.shape, "epochs:", epochs)
        
        # loop through the number of epochs
        e = 0
        print("Previous Epochs", self.n_epochs.eval(session=self.sess) )
        inc_epochs = self.n_epochs.assign(self.n_epochs + epochs)
        inc_epochs.op.run(session=self.sess)
        while ( e < epochs ):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            for i in range(0, len(ytrue_class), self.batch_size):
                # only train on factor of batch_size
                if i + self.batch_size >= len(ytrue_class):
                    continue
                # use feed dictionary with inputs and targets
                feed_dict_train = {
                                   self.target_class: ytrue_class[i:i+self.batch_size],
                                   self.input_class: x[i:i+self.batch_size]
                                   }
                
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
 
    # generate: generate a level given inputs
    # inputs
    # seed, matrix of level seed to generate next level
    # outputs:
    # y_generate, matrix of generated audio samples
    def generate(self, seed, sample_length):
        y_generate = np.append(seed, np.zeros(sample_length))
        print("Generating with seed:", seed.shape)
        print("Y generate", y_generate.shape )

        x_size = len(seed)
        y_size = len(y_generate)

        #print("X size", x_size)
        #print("y size", y_size)

        #print("Seed", y_generate[x_size - 100: x_size + 1])
        for i in range(x_size, y_size):
            #print("i:", i)
            #print("y_gen", i-self.batch_size, i )
            #print(y_generate[i-10:i])
            feed_dict_gen = {self.input_class: y_generate[i-self.batch_size:i] }
            y_g = self.sess.run( [self.prediction_value], feed_dict=feed_dict_gen)
            y_generate[i] = y_g[0][-1]
            #print("New value", y_g[0][-1])
        y_generate = raw(y_generate[x_size:])
        prev_epochs = self.n_epochs.eval(session=self.sess)
        print("Generated song:",  len(y_generate), "with Epochs", prev_epochs)
        return y_generate, prev_epochs

    # save: save neural network in specified self.data_location
    # inputs
    # close, bool value to specify if neural network should be closed
    # outputs:
    # none, saved neural network
    def save(self, close=False):
        self.saver.save(self.sess, self.save_file)
        print("Saving:", self.name)
        if close==True:
            self.sess.close()

    # load: loads neural network
    # inputs
    # none, requires a neural network chkpt file to be found in named directory
    # outputs:
    # none, runs save.restore() on the given nerual network
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

    # close: closes neural network safely within tensorflow
    # inputs
    # none, closes session
    # outputs:
    # none, session is closed
    def close(self):
        print("Closing" , self.name)
        self.sess.close()
