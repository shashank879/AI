import numpy as np
import tensorflow as tf
from ResNet import ResNet_5
from BoltzmannMachines import RestrictedBoltzmannMachine as RBM
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

# import dataset
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

# extract some data and reshape into proper form
train_x = mnist.train.images
train_y = mnist.train.labels

test_x = mnist.test.images
test_y = mnist.test.labels

# create model for this data

#ResNet
# train_x = train_x.reshape([-1, 28, 28, 1])
# test_x = test_x.reshape([-1, 28, 28, 1])
# net = ResNet_5()
# net.build_model(28, 28, 1, 10)

# net.train_model(train_x, train_y, name="resnet_5")
# net.test_model(test_x, test_y)
# net.eval_model(test_x[0:1,:,:,:])

#RBM
net = RBM(name="mnist_rbm")
n_features = train_x.shape[1]
net.build_model(n_features, 100)
# net.train_model(train_x)
net.load_model()

net.close()
print("Done")