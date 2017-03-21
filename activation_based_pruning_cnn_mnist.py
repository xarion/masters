# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 3
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 128  # 1st layer num features
n_hidden_2 = 256  # 2nd layer num features
n_input = 1  # MNIST data input (img shape: 28*28) # now it's 1 because conv2d
filter_size = 3
# tf Graph input (only pictures)
X = tf.placeholder("float", [batch_size, 784])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([filter_size, filter_size, n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([filter_size, filter_size, n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([filter_size, filter_size, n_hidden_1, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([filter_size, filter_size, n_input, n_hidden_1])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    x = tf.reshape(x, [batch_size, 28, 28, 1])
    with tf.name_scope("encoder_1"):
        # Encoder Hidden layer with sigmoid activation #1
        # Output: [?, 14, 14, 128]
        encoder_1 = tf.nn.relu(tf.add(tf.nn.conv2d(x, weights['encoder_h1'], strides=[1, 2, 2, 1], padding="SAME"),
                                      biases['encoder_b1']))

    with tf.name_scope("encoder_1"):
        # Decoder Hidden layer with sigmoid activation #2
        # Output: [?, 7, 7, 256]
        encoder_2 = tf.nn.relu(
            tf.add(tf.nn.conv2d(encoder_1, weights['encoder_h2'], strides=[1, 2, 2, 1], padding="SAME"),
                   biases['encoder_b2']))
    return encoder_2


# Building the decoder
def decoder(x):
    with tf.name_scope("decoder_1"):
        # Encoder Hidden layer with sigmoid activation #1
        #  Output: [?, 14, 14, 128]
        decoder_1 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(x, weights['decoder_h1'],
                                                             output_shape=[batch_size, 14, 14, n_hidden_1],
                                                             strides=[1, 2, 2, 1]), biases['decoder_b1']))
    with tf.name_scope("decoder_2"):
        # Decoder Hidden layer with sigmoid activation #2
        # Output: [?, 28, 28, 1]
        output = tf.add(tf.nn.conv2d_transpose(decoder_1, weights['decoder_h2'],
                                               output_shape=[batch_size, 28, 28, n_input],
                                               strides=[1, 2, 2, 1]), biases['decoder_b2'])
    return output


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = tf.reshape(decoder_op, [batch_size, 784])
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:batch_size]})
    # Compare original images with their reconstructions

    import matplotlib.pyplot as plt

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
