import tensorflow as tf
import numpy as np

def make_onehot(onehot_values, onehot_classes):
    onehot_matrix = np.zeros((onehot_values.size, onehot_classes))
    onehot_matrix[np.arange(onehot_values.size),onehot_values] = 1
    return onehot_matrix



class Model(object):
    def perceptron_nn(self, data, n_input_classes, n_target_classes, clip_size, 
                      n_nodes_hl1, n_nodes_hl2, n_nodes_hl3):

        hidden_1_layer = {'weights':tf.Variable(tf.random_normal(
                            [n_input_classes*clip_size, n_nodes_hl1])),
                          'biases':tf.Variable(tf.random_normal(
                            [n_nodes_hl1]))}

        hidden_2_layer = {'weights':tf.Variable(tf.random_normal(
                            [n_nodes_hl1, n_nodes_hl2])),
                          'biases':tf.Variable(tf.random_normal(
                            [n_nodes_hl2]))}

        hidden_3_layer = {'weights':tf.Variable(tf.random_normal(
                            [n_nodes_hl2, n_nodes_hl3])),
                          'biases':tf.Variable(tf.random_normal(
                            [n_nodes_hl3]))}


        output_layer = {'weights':tf.Variable(tf.random_normal(
                            [n_nodes_hl3, n_target_classes])),
                        'biases':tf.Variable(tf.random_normal(
                            [n_target_classes]))}


        l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
        return output

    def __init__(self, level, receptive_field,
                       batch_size=128, 
                       n_nodes_hl1=500,
                       n_nodes_hl2=500,
                       n_nodes_hl3=500):
        self.level = level
        self.batch_size = batch_size
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
        n_target_classes = 2**(level+2)



        inputs = tf.placeholder(tf.int32, [None,clip_size])
        onehot = tf.one_hot(inputs, n_input_classes)
        onehot = tf.reshape(onehot, (-1, clip_size*n_input_classes))
        onehot = tf.cast(onehot, tf.float32)

        target_class = tf.placeholder(tf.int64, [None])
        #target = tf.placeholder(tf.int64, [None, n_target_classes])
        
        target = tf.one_hot(target_class, n_target_classes)
        target = tf.reshape(target, (-1, n_target_classes))
        target = tf.cast(target, tf.float32)
        #target = tf.placeholder(tf.float32, [None, n_target_classes])

        logits = self.perceptron_nn(onehot, n_input_classes, n_target_classes, clip_size,
                                n_nodes_hl1, n_nodes_hl2, n_nodes_hl3)
        
        prediction = tf.nn.softmax(logits)
        prediction_class = tf.argmax(prediction, dimension=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        correct_prediction = tf.equal(target_class, prediction_class)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32) )

        sess = tf.Session()
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        self.clip_size = clip_size
        self.inputs = inputs
        self.target = target
        self.target_class = target_class
        self.onehot = onehot

        self.optimizer = optimizer
        self.cost = cost
        self.accuracy = accuracy

        self.sess = sess
        self.saver = saver
        self.save_path = "data/model_" + str(level) + ".ckpt"
        
        print self.save_path

    def train(self, x, ytrue_class, epochs=1 ):
        x = np.reshape(x, (-1, self.clip_size))
        ytrue_class = np.reshape(ytrue_class, (-1))
        print "Trainging:",  self.level, x.shape, ytrue_class.shape

        for e in range(epochs):
            epoch_loss = 0
            feed_dict_train = {self.inputs: x,
                               self.target_class: ytrue_class}
            _, c, a = self.sess.run([self.optimizer, self.cost, self.accuracy], feed_dict = feed_dict_train)
            
            epoch_loss+= c
            print "epoch", e, "loss", epoch_loss, "accuracy", a*100

    def save(self):
        self.saver.save(self.sess, self.save_path)
        print "Saving level", self.level, "at", self.save_path

    def load(self):
        self.saver.restore(self.sess, self.save_path)
        print "Loading level", self.level

    def close(self):
        self.sess.close()
