import numpy as np
import tensorflow as tf
import Utilities as utils
from pathlib import Path

EPOCHS = 5
BATCH_SIZE = 100

class ResNet():

    def __init__(self, name=None):
        self.model = None
        self.name = name
        self.sess = tf.Session()
        print("Session started")

    def close(self):
        self.sess.close()
        print("Session closed")

    def highway(self, x, in_channels, out_channels, stride, name=None):
        if in_channels != out_channels:
            return utils.conv2d(x, 1, in_channels, out_channels, stride, name+"_conv")
        else:
            return tf.identity(x)

    def basic_unit(self, x, conv_size, channels, stride, name=None):
        y = utils.batch_normalise(x)
        y = tf.nn.relu(y)
        y = utils.conv2d(y, conv_size, self.channels, channels, stride, name+"_conv1")
        y = utils.batch_normalise(y)
        y = tf.nn.relu(y)
        y = utils.conv2d(y, conv_size, channels, channels, 1, name+"_conv2")
        shortcut = self.highway(x, self.channels, channels, stride, name+"_highway")
        y = tf.add(shortcut, y)
        self.channels = channels
        return y

    def save_path(self):
        return "./saved/" + self.name + ".ckpt"

    def load_model(self):
        path = self.save_path()
        file = Path(path)
        if file.is_file():
            tf.train.Saver().restore(self.sess, path)
            print("Model restored from path... ", path)

    def train_model(self, data, outputs, name=None):
        prediction = self.model
        sess = self.sess
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        sess.run(tf.initialize_all_variables())
        samples = np.shape(data)[0]
        for epoch in range(EPOCHS):
            epoch_loss = 0
            nBatches = int(samples/BATCH_SIZE)
            for i in range(nBatches):
                epoch_x = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :, :]
                epoch_y = outputs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                _, c = sess.run([optimizer, cost], feed_dict={self.x:epoch_x, self.y:epoch_y})
                epoch_loss += c
                    
                print('Epoch progress... {0}%'.format(int(i/nBatches*100)), end='\r')    
            print('Epoch: ', epoch+1, ', Loss: ', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(outputs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Calculating accuracy on the training set...")
        print("Train Accuracy: ", accuracy.eval(feed_dict={self.x:data, self.y:outputs}, session=self.sess))

        if name != None:
            save_path = tf.train.Saver().save(sess, self.save_path())
            print("Model saved at... ", save_path)

    def test_model(self, data, outputs):
        prediction = self.model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(outputs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Calculating accuracy on the testing set...")
        print("Test Accuracy: ", accuracy.eval(feed_dict={self.x:data, self.y:outputs}, session=self.sess))
        
    def eval_model(self, data):
        prediction = self.model
        output = tf.cast(tf.argmax(prediction, 1), 'float')
        print("Class: ", output.eval(feed_dict={self.x:data}, session=self.sess))

class ResNet_5(ResNet):

    def build_model(self, width, height, nChannels, nClasses):
        # Input and Output Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, width, height, nChannels])
        self.y = tf.placeholder(tf.float32, shape=[None, nClasses])
        self.channels = nChannels

        # Create Model

        # Layer 1
        model = utils.conv2d(self.x, 3, self.channels, 16, 1, name="layer_1_conv")
        self.channels = 16
        model = utils.batch_normalise(model)
        model = tf.nn.relu(model)

        # Layer 2:4
        model = self.basic_unit(model, 3, 16, 1, name="layer_2")
        model = self.basic_unit(model, 3, 32, 2, name="layer_3")
        model = self.basic_unit(model, 3, 64, 2, name="layer_4")

        # Layer 5
        side = model.get_shape()[1]
        model = utils.avg_pool(model, side, 1, padding='VALID')
        model = tf.reshape(model, [-1, 64])
        W = utils.weight_variable([64, nClasses], "layer_5_w")
        model =  tf.add(tf.matmul(model, W), utils.bias_variable([nClasses], "layer_5_b"))

        # Save for later
        self.model = model

        return model
