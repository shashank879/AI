import numpy as np
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool(x, size, stride):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')

def avg_pool(x, size, stride):
    return tf.nn.avg_pool(x, ksize=size, strides=stride, padding='SAME')

def batch_normalise(x):
    x_mean, x_var = tf.nn.moments(x, [0])
    print (x_mean, x_var)
    return tf.nn.batch_normalization(x, x_mean, x_var, None, None, variance_epsilon=1e-3)
        
class ResNet_v1_18:

    def add_basic_unit(self, x, conv_size, channels, stride):
        y = batch_normalise(x)
        y = tf.nn.relu(x)
        w1 = weight_variable([conv_size, conv_size, self.channels, channels])
        b1 = bias_variable([channels])
        y = tf.add( conv2d(y, w1, stride), b1 )
        y = batch_normalise(y)
        y = tf.nn.relu(y)
        w2 = weight_variable([conv_size, conv_size, channels, channels])
        b2 = bias_variable([channels])
        y = tf.add( conv2d(y, w2, stride), b2 )
        x = tf.add(x, y)
        self.channels = channels
        return x

    def build_and_train_model(self, x, y):
        size = np.shape(x)
        data = tf.placeholder(tf.float32, shape=[None, size[1], size[2], size[3]])
        self.channels = size[3]
        model = self.add_basic_unit(data, 7, 32, 1)
        model = self.add_basic_unit(model, 3, 32, 1)

        self.model = model

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            sess.run(model, feed_dict={data:x})
