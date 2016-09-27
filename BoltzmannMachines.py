import numpy as np
import tensorflow as tf
import Utilities as utils
from pathlib import Path

LEARN_RATE = 0.0001
BATCH_SIZE = 10
EPOCHS = 10

def sample_from_prb(prb, random):
    sign = tf.sign(prb - random)
    return tf.nn.relu(sign)

class RestrictedBoltzmannMachine:

    def __init__(self, gibbs_steps=1, visible_units_type='bin', name=None):
        self.visible_units_type = visible_units_type
        self.gibbs_steps = gibbs_steps
        self.name = name

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
        self.cost = None

        self.sess = tf.Session()

    def close(self):
        self.sess.close()

    def sample_hidden_from_visible(self, visible):
        prb = tf.nn.sigmoid(tf.add(tf.matmul(visible, self.w), self.h_b))
        sample = sample_from_prb(prb, self.h_rand)
        return prb, sample

    def sample_visible_from_hidden(self, hidden):
        act = tf.matmul(hidden, tf.transpose(self.w)) + self.v_b
        prb = None
        if self.visible_units_type == 'bin':
            prb = tf.nn.sigmoid(act)
        elif self.visible_units_type == 'gauss':
            prb = tf.truncated_normal(shape=[1, self.n_features], mean=act, stddev=1.0)
        
        sample = sample_from_prb(prb, self.v_rand)
        return prb, sample

    def gibbs_step_vhv(self, visible):
        h0_prb, h0_samples = self.sample_hidden_from_visible(visible)
        v_prb, _ = self.sample_visible_from_hidden(h0_prb)
        h1_prb, h1_samples = self.sample_hidden_from_visible(v_prb)
        return h0_prb, h0_samples, v_prb, h1_prb, h1_samples
    
    def gibbs_step_hvh(self, hidden):
        v0_prb, v0_samples = self.sample_visible_from_hidden(hidden)
        h_prb, _ = self.sample_hidden_from_visible(v0_prb)
        v1_prb, v1_samples = self.sample_visible_from_hidden(h_prb)
        return v0_prb, v0_samples, h_prb, v1_prb, v1_samples

    def positive(self, visible, hidden_prb, hidden_samples):
        if self.visible_units_type == 'bin':
            return tf.matmul(tf.transpose(visible), hidden_samples)
        elif self.visible_units_type == 'gauss':
            return tf.matmul(tf.transpose(visible), hidden_prb)

        return None

    def save_path(self):
        return "./saved/" + self.name + ".ckpt"

    def load_model(self):
        path = self.save_path()
        file = Path(path)
        if file.is_file():
            tf.train.Saver().restore(self.sess, path)
            print("Model loaded from path... " + path)

    def build_model(self, n_features, n_hidden):
        print("Building model...")
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
        self.w = utils.weight_variable([n_features, n_hidden], stddev=0.01, name="weights")
        self.v_b = utils.bias_variable([n_features], value=0, name="visible_biases")
        self.h_b = utils.bias_variable([n_hidden], value=0, name="hidden_biases")

        # Calculate 1st gibbs step
        h0_prb, h0_samples, v_prb, h1_prb, h1_samples = self.gibbs_step_vhv(self.visible_units)
        # 1st part of the CD
        positive = self.positive(self.visible_units, h0_prb, h0_samples)

        # Calculate the remaining gibbs steps
        nn_input = v_prb
        for step in range(self.gibbs_steps - 1):
            h_prb, h_samples, v_prb, h1_prb, h1_samples = self.gibbs_step_vhv(nn_input)
            nn_input = v_prb
        
        # 2nd part of the CD
        negative = tf.matmul(tf.transpose(v_prb), h1_prb)

        # Update rules for the weights and biases
        self.w_update = self.w.assign_add(LEARN_RATE * (positive - negative) / BATCH_SIZE)
        self.h_b_update = self.h_b.assign_add(LEARN_RATE * tf.reduce_mean(h0_prb - h1_prb, 0))
        self.v_b_update = self.v_b.assign_add(LEARN_RATE * tf.reduce_mean(self.visible_units - v_prb, 0))

        # Mean squared error for reconstruction cost
        self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.visible_units - v_prb)))

        print("Model ready...")
        return self.visible_units, self.hidden_units

    def train_model(self, data, batch_size=10):
        print("Training model...")
        self.sess.run(tf.initialize_all_variables())
        np.random.shuffle(data)
        nBatches = int(data.shape[0]/BATCH_SIZE)
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for i in range(nBatches): 
                epoch_data = data[BATCH_SIZE*i:BATCH_SIZE*(i+1),:]
                v_rand = np.random.rand(epoch_data.shape[0], self.n_features)
                h_rand = np.random.rand(epoch_data.shape[0], self.n_hidden)
                feed_dict={self.visible_units: epoch_data,
                           self.v_rand: v_rand,
                           self.h_rand: h_rand}
                _,_,_,cost = self.sess.run([self.w_update, self.v_b_update, self.h_b_update, self.cost], feed_dict=feed_dict)
                epoch_loss += cost
                print('Epoch progress... {0}%'.format(int(i/nBatches*100)), end='\r')
            print('Epoch: ', epoch+1, ', Reconstruction error: ', epoch_loss)
        print("Model trained...")

        if self.name != None:
            saver = tf.train.Saver()
            path = saver.save(self.sess, self.save_path())
            print("Model saved at... ", path)

    def eval_visible(self, hidden_states, gibbs_steps):
        v0_prb, v0_samples, h_prb, v1_prb, v1_samples = self.gibbs_step_hvh(self.hidden_units)
        # For sampling from a trained RBM
        for step in range(gibbs_steps-1):
            v0_prb, v0_samples, h_prb, v1_prb, v1_samples = self.gibbs_step_hvh(h_prb)
            
        error = tf.sqrt(tf.reduce_mean(tf.square(self.hidden_units - h_prb), 0))
        v_rand = np.random.rand(hidden_states.shape[0], self.n_features)
        feed_dict={self.hidden_units : hidden_states,
                   self.v_rand : v_rand}
        return self.sess.run([v0_prb, error], feed_dict=feed_dict)

    def eval_hidden(self, visible_states):
        self.hidden,_ = self.sample_hidden_from_visible(self.visible_units)
        h_rand = np.random.rand(visible_states.shape[0], self.n_hidden)
        feed_dict={self.visible_units : visible_states, 
                   self.h_rand : h_rand}
        prb,states = self.sess.run(self.hidden, feed_dict=feed_dict)
        return states

    def show_weight_graph(self):
        w = self.w.eval()

# Test run
# rbm = RestrictedBoltzmannMachine()
# rbm.build_model(3,2)
# data = np.array([[0,0,1],[0,1,1],[1,0,0],[1,1,1]])
# rbm.train_model(data)
# rbm.show()
# rbm.close()

# print("Done")