import numpy as np
import tensorflow as tf
import networkx as nx
import Utilities as utils

LEARN_RATE = 0.01
BATCH_SIZE = 10
EPOCHS = 5

def sample_from_prb(prb, random):
    sign = tf.sign(prb - random)
    return tf.nn.relu(sign)

class RestrictedBoltzmannMachine:

    def __init__(self, gibbs_steps=1, visible_units_type='bin'):
        self.visible_units_type = visible_units_type
        self.gibbs_steps = gibbs_steps

        self.visible_units = None
        self.hidden_units = None
        self.w = None
        self.v_b = None
        self.h_b = None
        self.v_rand = None
        self.h_rand = None

        self.w_update = None
        self.h_b_update = None
        self.v_b_update = None

        self.sess = tf.Session()

    def sample_hidden(self, visible):
        prb = tf.nn.sigmoid(tf.add(tf.matmul(visible, self.w), self.h_b))
        sample = sample_from_prb(prb, self.h_rand)
        return prb, sample

    def sample_visible(self, hidden):
        act = tf.matmul(hidden, tf.transpose(self.w)) + self.v_b

        if self.visible_units_type == 'bin':
            return tf.nn.sigmoid(act)
        elif self.visible_units_type == 'gauss':
            return tf.truncated_normal(shape=[1, self.n_features], mean=act, stddev=1.0)

        return None

    def gibbs_step_vhv(self, visible):
        h0_prb, h0_samples = self.sample_hidden(visible)
        v_prb = self.sample_visible(h0_prb)
        h1_prb, h1_samples =  self.sample_hidden(v_prb)
        return h0_prb, h0_samples, v_prb, h1_prb 

    def positive(self, visible, hidden_prb, hidden_samples):
        if self.visible_units_type == 'bin':
            return tf.matmul(tf.transpose(visible), hidden_samples)
        elif self.visible_units_type == 'gauss':
            return tf.matmul(tf.transpose(visible), hidden_prb)

        return None

    def train_model(self, data, batch_size=10):
        np.random.shuffle(data)
        nBatches = int(data.shape[0]/BATCH_SIZE)
        for epoch in range(EPOCHS):
            for i in range(nBatches): 
                epoch_data = data[epoch*i:epoch*(i+1),:,:,:]
                v_rand = np.random.rand([epoch_data.shape[0], self.n_features])
                h_rand = np.random.rand([epoch_data.shape[0], self.n_hidden])
                feed_dict={self.visible_units: epoch_data,
                           self.v_rand: v_rand,
                           self.h_rand: h_rand}
                self.sess.run([self.w_update, self.v_b_update, self.h_b_update], feed_dict=feed_dict)

        print("Model trained...")

    def build_model(self, n_features, n_hidden):

        self.n_features = n_features
        self.n_hidden = n_hidden

        """ make placeholders for data
            v = visible units
            h = hidden units
        """
        self.visible_units = tf.placeholder('float32', [None, n_features], "Input")
        self.hidden_units = tf.placeholder('float32', [None, n_hidden], "Hidden")
        self.v_rand = tf.placeholder('float32', [None, n_features])
        self.h_rand = tf.placeholder('float32', [None, n_hidden])
        
        """ create required set of variables
            w = Weight matrix (v,h)
            v_b = Visible biases (v)
            h_b = Hidden biases (h)
        """
        self.w = utils.weight_variable([n_features, n_hidden], name="weights")
        self.v_b = utils.bias_variable([n_features], name="visible_biases")
        self.h_b = utils.bias_variable([n_hidden], name="hidden_biases")

        h0_prb, h0_samples, v_prb, h1_prb = self.gibbs_step_vhv(self.visible_units)
        positive = self.positive(self.visible_units, h0_prb, h0_samples)

        for step in range(self.gibbs_steps - 1):
            h_prb, h_samples, v_prb, h1_prb = self.gibbs_step_vhv(v_prb)
        
        negative = tf.matmul(tf.transpose(v_prb), h1_prb)

        self.w_update = self.w.assign_add(LEARN_RATE * (positive - negative) / BATCH_SIZE)
        self.h_b_update = self.h_b.assign_add(LEARN_RATE * tf.reduce_mean(h0_prb - h1_prb, 0))
        self.v_b_update = self.v_b.assign_add(LEARN_RATE * tf.reduce_mean(self.visible_units - v_prb, 0))

        print("Model ready...")

rbm = RestrictedBoltzmannMachine()
rbm.build_model(3,2)
data = np.array([[0,0,1],[0,1,1],[1,0,0],[1,1,1]])
rbm.train_model(data)