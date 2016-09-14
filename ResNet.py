import numpy as np
import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 100

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, conv_size, in_channels, out_channels, stride):
    weights = weight_variable([conv_size, conv_size, in_channels, out_channels])
    biases = bias_variable([out_channels])
    conv = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding='SAME')
    return tf.add(conv, biases)

def max_pool(x, size, stride):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')

def avg_pool(x, size, stride):
    return tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')

def batch_normalise(x):
    x_mean, x_var = tf.nn.moments(x, [0])
    print (x_mean, x_var)
    return tf.nn.batch_normalization(x, x_mean, x_var, None, None, variance_epsilon=1e-3)

class ResNet:

    def highway(self, x, in_channels, out_channels, stride):
        if in_channels != out_channels:
            return conv2d(x, 1, in_channels, out_channels, stride)
        else:
            return tf.identity(x)

    def basic_unit(self, x, conv_size, channels, stride):
        y = batch_normalise(x)
        y = tf.nn.relu(y)
        y = conv2d(y, conv_size, self.channels, channels, stride)
        y = batch_normalise(y)
        y = tf.nn.relu(y)
        y = conv2d(y, conv_size, channels, channels, 1)
        shortcut = self.highway(x, self.channels, channels, stride)
        y = tf.add(shortcut, y)
        self.channels = channels
        return y

    def build_model(self, width, height, channels):
        self.model = None

    def train_model(self, data, outputs):
        prediction = self.model
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, outputs))

        optimizer = tf.train.AdamOptimizer().minimize(cost)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            samples = tf.size(data)[0]
            for epoch in range(EPOCHS):
                epoch_loss = 0
                for i in range(samples/BATCH_SIZE):
                    epoch_x = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :, :, :]
                    epoch_y = outputs[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                    _, c = sess.run([optimizer, cost], feed_dict={self.x:epoch_x, self.y:epoch_y})
                    epoch_loss += c
                
                print('Epoch: ', epoch+1, ', Loss: ', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(outputs, 1))
            accuracy = tf.reduce_mean(tf.cast(corrcect, 'float'))
            print("Train Accuracy: ", accuracy.eval(feed_dict={self.x:data, self.y:outputs}))

    def test_model(self, data, outputs):
        prediction = self.model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(outputs, 1))
        accuracy = tf.reduce_mean(tf.cast(corrcect, 'float'))
        print("Test Accuracy: ", accuracy.eval(feed_dict={self.x:data, self.y:outputs}))
        
    def eval_model(self, data):
        prediction = self.model
        output = prediction.eval(feed_dict={self.x:data})
        print(output)

class ResNet_20(ResNet):

    def build_model(self, width, height, nChannels, nClasses):
        # Input and Output Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, width, height, nChannels])
        self.y = tf.placeholder(tf.float32, shape=[None, nClasses])
        self.channels = nChannels

        # Create Model
        model = conv2d(self.x, 3, self.channels, 16, 1)
        self.channels = 16
        model = batch_normalise(model)
        model = tf.nn.relu(model)
        model = self.basic_unit(model, 3, 16, 1)
        model = self.basic_unit(model, 3, 32, 2)
        model = self.basic_unit(model, 3, 64, 2)
        model = avg_pool(model, 7, 1)
        model = tf.reshape(model, [-1, 7*7, 1])
        W = weight_variable([256, nClasses])
        model =  tf.add(tf.matmul(model, W) + bias_variable([None, nClasses]))

        # Save for later
        self.model = model

        return model
