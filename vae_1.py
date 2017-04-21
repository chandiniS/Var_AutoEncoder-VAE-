import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# get the data
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets("/tmp/data", one_hot=True)  # one hot encoding is needed

num_pixel = 28 * 28  # std mnist image dimension and tensor parameters

# Unsupervised learning- dont care about labels
X = tf.placeholder(tf.float32, shape=([None, num_pixel]))


# few functions for generating weights and biases- reused in every layer

def weights_init(shape, name):
    initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
    return initial


def bias_init(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

# a fully connected layer- we want the entire image, to generate a new image


def FC_layer(X,W,b):
    return tf.matmul(X, W)+b

latent_dim = 20 # need to try different values
hidden_dim = 500 # number of units per hidden layers

#layer 1
w_enc = weights_init([num_pixel, hidden_dim], "Wenc_1")
b_enc = bias_init([hidden_dim], "benc_1")

h_enc = tf.nn.tanh(FC_layer(X, w_enc, b_enc))  #Activation tanh - range between -1 & 1 avoids vanishing gradient problem

#layer 2
w_mu = weights_init([hidden_dim, latent_dim], "Wenc_2")
b_mu = bias_init([latent_dim], "benc_2")

mu = FC_layer(h_enc, w_mu, b_mu)

#Reparameterization step- output is not a vector of actual value
#we backpropagate the mean and standard deviations of output

#standard deviation layer- log sigma
w_logstd = weights_init([hidden_dim, latent_dim], "Wlogstd")
b_logstd = bias_init([latent_dim], "blogstd")

logstd = FC_layer(h_enc,w_logstd,b_logstd)

# Important step
noise = tf.random_normal([1, latent_dim])

#Z is the final output of the encoder

Z = mu + tf.multiply(noise, tf.exp(0.5 * logstd))  # tf.mul has renamed to tf.multiply in version- this is .*

# DECODER
# layer1


w_dec = weights_init([latent_dim, hidden_dim], "Wdec_1")
b_dec = bias_init([hidden_dim], "bdec_1")

h_dec = tf.nn.tanh(FC_layer(Z, w_dec,b_dec))

# layer 2
w_recon = weights_init([hidden_dim,num_pixel], "Wdec_2")
b_recon = bias_init([num_pixel], "bdec_2")

# outputs a 784 pixel -sigmoid fro o/p 0 or 1

op_recon = tf.nn.sigmoid(FC_layer(h_dec,w_recon,b_recon))

# Loss Function -Compare generated image to actual image
#Log likelihood - KLDivergence. Here KL divergence is regularization
llh = tf.reduce_sum(X * tf.log(op_recon + 1e-9) + (1 - X)*tf.log(1 - op_recon + 1e-9),reduction_indices=1)

KL_Div = -0.5*tf.reduce_sum(1 + 2*logstd - tf.pow(mu, 2) -tf.exp(2*logstd), reduction_indices=1)

#Variational lower bound- which is the
v_lb =
