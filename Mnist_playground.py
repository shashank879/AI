import numpy as np
import tensorflow as tf
from ResNet import ResNet_5
from BoltzmannMachines import RestrictedBoltzmannMachine as RBM
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import Utilities as utils

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
# train_x = np.sign(train_x - 0.2)
# train_x = np.maximum(train_x, 0)
# y = np.argmax(train_y, 1)
# y = np.reshape(y,[y.shape[0],1])
# data = np.concatenate((train_x, y), axis=1)
# data = np.where(data[:,784] == 5)
# train_x = data[:,0:783]
n_features = train_x.shape[1]
n_hidden = 200
net = RBM(gibbs_steps=5, name=("mnist_rbm" + str(n_hidden)))
net.build_model(n_features, n_hidden)
# net.train_model(train_x)
net.load_model()
# net.show()

y = np.array([[i==j for j in range(n_hidden)] for i in range(n_hidden)], dtype='float32')
y, error = net.eval_visible(y, 100)
print("Variation", np.mean(np.var(y,0)))
print("Reconstruction error... ", np.mean(error))
imgs = np.reshape(y, [n_hidden, 28, 28])
imgs = imgs*255
img_C = utils.combine_images(imgs, 20)
img_C.show()

# rep = net.eval_hidden(test_x[0:2000,:])
# labels = np.argmax(test_y[0:2000,:], 1)
# utils.tsne2D(rep,labels=labels)

net.close()
print("Done")
