import tensorflow as tf
import numpy as np
import os


def nn_layer(input_layer, n_nodes_in, n_nodes, output_layer=False):
    hl_weight = tf.Variable(tf.random_normal([n_nodes_in, n_nodes]))
    hl_bias = tf.Variable(tf.random_normal([n_nodes]))
    if(not output_layer):
        layer = tf.add(tf.matmul(input_layer, hl_weight), hl_bias)
        layer = tf.nn.relu(layer)
    else:
        layer =  tf.matmul(input_layer, hl_weight) + hl_bias
    return layer

class Model(object):
    def perceptron_nn(self, data, n_input_classes, n_target_classes, clip_size, n_nodes):
        layers = []
        for i, nodes in enumerate( n_nodes ):
            if i == 0:
                layers.append(nn_layer(data, n_input_classes*clip_size, n_nodes[0]))
            else:
                layers.append(nn_layer(layers[i-1], n_nodes[i-1], n_nodes[i]))
        output_layer = nn_layer(layers[-1], n_nodes[-1], n_target_classes, output_layer=True)
        return output_layer

    def __init__(self, level, receptive_field, data_location, 
                 batch_size=128, normalize_mode=True ):
        self.level = level
        self.batch_size = batch_size
        self.receptive_field = receptive_field
        '''
        self.x_name = x_name
        self.ytrue_name = ytrue_name
        self.batch_size = batch_size
        self.n_nodes_hl1 = n_nodes_hl1
        self.n_nodes_hl2 = n_nodes_hl2
        self.n_nodes_hl3 = n_nodes_hl3
        '''

        clip_size = 2*receptive_field+1
        n_input_classes = 2**(level+1)
        input_classes_max = n_input_classes - 1
        n_target_classes = 2**(level+2)
        target_classes_max = n_target_classes - 1



        inputs = tf.placeholder(tf.int64, [None,clip_size])
        target_class = tf.placeholder(tf.int64, [None])

        #create onehot value for non-normalized inputs
        onehot = tf.one_hot(inputs, n_input_classes)
        onehot = tf.reshape(onehot, (-1, clip_size*n_input_classes))
        onehot = tf.cast(onehot, tf.float32)
        # create regular onehot values for target
        target = tf.one_hot(target_class, n_target_classes)
        target = tf.reshape(target, (-1, n_target_classes))
        target = tf.cast(target, tf.float32)

        #normalized inputs and target, along with corresponding onehot

        # slices tensor from middle value -> [0, middle_index] 
        # to end of None -> [-1(end), 1 (one value only)] 
        middle = tf.slice(inputs, [0,  receptive_field] , [-1, 1])
        middle_ = tf.reshape(middle, [-1] )
        normalized = tf.subtract(inputs, middle)
        normalized_pos = normalized + input_classes_max
         
        input_norm_onehot_range = input_classes_max*2+1
        normalized_onehot = tf.one_hot(normalized_pos, input_norm_onehot_range)
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
        n_nodes = [ (level+1)*400, (level+1)*100, (level+1)*50, ]


        if normalize_mode == True:
            nn_inputs = normalized_onehot
            nn_targets = target_normalized_onehot
            nn_target_class = target_normalize_pos_
            nn_n_inputs = input_norm_onehot_range
            nn_n_targets = target_norm_onehot_range
        else:
            nn_inputs = onehot
            nn_targets = target
            nn_target_class = target_class
            nn_n_inputs = n_input_classes
            nn_n_targets = n_target_classes

        logits = self.perceptron_nn(nn_inputs, nn_n_inputs, nn_n_targets, clip_size, n_nodes)
        
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
        
        self.middle = middle
        self.normalized = normalized
        self.normalized_pos = normalized_pos
        self.normalized_onehot = normalized_onehot

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
        
        self.name = "model_" + str(level) + "_r" + str(receptive_field)
        self.save_dir = data_location + "/" + self.name
        self.save_file = self.save_dir + "/" + self.name + ".ckpt"

        if( os.path.isdir(self.save_dir) ):
            print "Loading previous:", self.name
            self.load()
        else:
            os.makedirs( self.save_dir )
            print "Creating level directory at:", self.save_dir
        print self.save_file

    def test_io_onehot(self, x, ytrue_class):
        np.set_printoptions(threshold=np.inf)
        #x = np.reshape(x, (-1, self.clip_size))
        ytrue_class = np.reshape(ytrue_class, (-1))
        print "Testing:",  self.name, x.shape, ytrue_class.shape
        for i in range(len(x)/2, len(x)/2 + 1, 10):
            feed_dict_test = {self.inputs: x[i:i+10,:],
                               self.target_class: ytrue_class[i:i+10] }
            inp, mid, norm, norm_pos, norm_one, one, targ_c, in_scale, tar_nor, tar_nor_pos, tar_nor_pos_one, tar = self.sess.run(
                    [self.inputs, self.middle, self.normalized, self.normalized_pos, 
                     self.normalized_onehot, self.onehot, self.target_class, self.inputs_scaled,
                     self.target_normalized, self.target_normalized_pos, 
                     self.target_normalized_onehot, self.target],
                    feed_dict=feed_dict_test)
            print "inputs\n", inp
            print "middle value of receptive_field\n", mid
            print "normalized inputs\n", norm
            print "positive normalized\n", norm_pos
            print "onehot normalized inputs\n", norm_one
            print "regular onehot inputs\n", one
            
            print "inputs scaled to next level\n", in_scale
            print "target classes\n", targ_c
            print "targets normalized\n", tar_nor
            print "positive normalized targets\n", tar_nor_pos
            print "onehot normalized targets\n", tar_nor_pos_one
            print "regular onehot targets\n", tar


    def train(self, x, ytrue_class, epochs=1 ):
        #x = np.reshape(x, (-1, self.clip_size))
        ytrue_class = np.reshape(ytrue_class, (-1))
        print "Trainging:",  self.name, x.shape, ytrue_class.shape, "epochs:", epochs
        
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

            print "epoch", e, "loss", epoch_loss
            e = e+1

            if (epoch_total != 0):
                epoch_accuracy = 100.0 * float(epoch_correct) / float(epoch_total)
                print "accuracy:", epoch_accuracy
                if epoch_accuracy > self.best_accuracy :
                    self.best_accuracy = epoch_accuracy

    def generate(self, x_seed):
        x_seed = np.reshape(x_seed, (-1, self.clip_size))
        print "Generating with seed:", x_seed.shape
        y_generated = []
        for i in range(0, len(x_seed), self.batch_size):
            feed_dict_gen = {self.inputs: x_seed[i:i+self.batch_size,:]}
            y_g = self.sess.run( [self.prediction_value], feed_dict=feed_dict_gen)
            y_generated = np.append(y_generated, y_g)
        print "Generated song:",  len(y_generated)
        return y_generated

    def save(self, close=False):
        self.saver.save(self.sess, self.save_file)
        print "Saving:", self.name
        if close==True:
            self.sess.close()

    def load(self):
        if os.path.isdir(self.save_dir):
            print "Loading:", self.name, self.save_file
            self.saver.restore(self.sess, self.save_file)
            return True
        else:
            print "Failed loading:", self.name
            return False

    def close(self):
        print "Closing" , self.name
        self.sess.close()
