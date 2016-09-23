import tensorflow as tf

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
    