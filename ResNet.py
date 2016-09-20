import numpy as np
import tensorflow as tf

EPOCHS = 5
BATCH_SIZE = 100

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

def conv2d(x, conv_size, in_channels, out_channels, stride, name=None):
    weights = weight_variable([conv_size, conv_size, in_channels, out_channels], name=name+"_w")
    biases = bias_variable([out_channels], name=name+"_b")
    conv = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding='SAME')
    return tf.add(conv, biases)

def max_pool(x, size, stride):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')

def avg_pool(x, size, stride, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding)

def batch_normalise(x):
    x_mean, x_var = tf.nn.moments(x, [0])
    print (x_mean, x_var)
    return tf.nn.batch_normalization(x, x_mean, x_var, None, None, variance_epsilon=1e-3)

class ResNet():

    def __init__(self):
        self.model = None
        self.sess = tf.Session()
        print("Session started")

    def close(self):
        self.sess.close()
        print("Session closed")

    def highway(self, x, in_channels, out_channels, stride, name=None):
        if in_channels != out_channels:
            return conv2d(x, 1, in_channels, out_channels, stride, name+"_conv")
        else:
            return tf.identity(x)

    def basic_unit(self, x, conv_size, channels, stride, name=None):
        y = batch_normalise(x)
        y = tf.nn.relu(y)
        y = conv2d(y, conv_size, self.channels, channels, stride, name+"_conv1")
        y = batch_normalise(y)
        y = tf.nn.relu(y)
        y = conv2d(y, conv_size, channels, channels, 1, name+"_conv2")
        shortcut = self.highway(x, self.channels, channels, stride, name+"_highway")
        y = tf.add(shortcut, y)
        self.channels = channels
        return y

    def load_model(self, name):
        path = "./models/"+name+".ckpt"
        tf.train.Saver().restore(self.sess, path)
        print("Model restored...")

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
            print("Saving model...")
            path = "./models/"+name+".ckpt"
            save_path = tf.train.Saver().save(sess, path)
            print("Model saved at", save_path)

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
        model = conv2d(self.x, 3, self.channels, 16, 1, name="layer_1_conv")
        self.channels = 16
        model = batch_normalise(model)
        model = tf.nn.relu(model)

        # Layer 2:4
        model = self.basic_unit(model, 3, 16, 1, name="layer_2")
        model = self.basic_unit(model, 3, 32, 2, name="layer_3")
        model = self.basic_unit(model, 3, 64, 2, name="layer_4")

        # Layer 5
        side = model.get_shape()[1]
        model = avg_pool(model, side, 1, padding='VALID')
        model = tf.reshape(model, [-1, 64])
        W = weight_variable([64, nClasses], "layer_5_w")
        model =  tf.add(tf.matmul(model, W), bias_variable([nClasses], "layer_5_b"))

        # Save for later
        self.model = model

        return model
