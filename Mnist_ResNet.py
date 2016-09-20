import numpy as np
import tensorflow as tf
from ResNet import ResNet_5
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
net = ResNet_5()
net.build_model(28, 28, 1, 10)

net.train_model(train_x[0:100,:,:,:], train_y[0:100,:], name="resnet_5")
net.test_model(test_x, test_y)
net.eval_model(test_x[0:1,:,:,:])

net.close()
print("Done")