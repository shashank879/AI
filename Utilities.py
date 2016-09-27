import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from PIL import Image

def weight_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev, name=name)
    return tf.Variable(initial)

def bias_variable(shape, value=0.1, name=None):
    initial = tf.constant(value=value, dtype='float32', shape=shape, name=name)
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

def tsne2D(data, labels=None):
    tsne = TSNE()
    out = tsne.fit_transform(data)
    plt.scatter(out[:,0], out[:,1])
    plt.show()

def combine_images(imgs, picturesPerRow = 16):
    shape = imgs[0].shape
    n = imgs.shape[0]
    rows = int(n/picturesPerRow) + 1*((n%picturesPerRow)!=0)
    img_C = Image.new('F', size=(shape[1] * picturesPerRow, shape[0] * rows))

    z = 0
    for j in range(0, rows * shape[1], shape[1]):
        for i in range(0, picturesPerRow * shape[0], shape[0]):
            img = Image.fromarray(imgs[z])
            img_C.paste(im=img, box=(i,j,i+shape[0],j+shape[1]))
            z+=1
            if z >= n:
                return img_C

    return img_C
