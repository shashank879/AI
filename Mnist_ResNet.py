import numpy as np
import tensorflow as tf
from ResNet import ResNet_v1_18
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

# import dataset
mnist = input_data.read_data_sets("./data/mnist", one_hot=True)

# extract some data and reshape into proper form
x, y = mnist.train.next_batch(100)
x = x.reshape([-1, 28, 28, 1])

# create model for this data
model = ResNet_v1_18()
model.build_and_train_model(x, y)

print("Done")