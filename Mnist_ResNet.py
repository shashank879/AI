import numpy as np
import tensorflow as tf
from ResNet import ResNet_20
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

# import dataset
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

# extract some data and reshape into proper form
train_x = mnist.train.images
train_y = mnist.train.labels
train_x = train_x.reshape([-1, 28, 28, 1])

test_x = mnist.test.images
test_y = mnist.test.labels
test_x = test_x.reshape([-1, 28, 28, 1])

# create model for this data
net = ResNet_20()
net.build_model(28, 28, 1, 10)

net.train_model(train_x, train_y)
net.test_model(test_x, test_y)
net.eval_model(test_x[0:1,:,:,:])

net.close()
print("Done")