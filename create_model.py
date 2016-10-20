import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

learning_rate = 0.01
training_epochs = 100
display_step = 50

# Training Data
train_X = np.random.beta(0.5, 5, size=[100])
train_Y = np.random.beta(0.1, 1, size=[100])
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder(shape=[1, 1], dtype=tf.float32)
Y = tf.placeholder(shape=[1, 1], dtype=tf.float32)

W1 = tf.Variable(tf.random_normal([1, 32]), name="l1_weights", dtype=tf.float32)
b1 = tf.Variable(tf.random_normal([32]), name="l1_bias", dtype=tf.float32)

W2 = tf.Variable(tf.random_normal([32, 1]), name="out_weights", dtype=tf.float32)
b2 = tf.Variable(tf.random_normal([1]), name="out_bias", dtype=tf.float32)

l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(X, W1), b1, name="l1"))
pred = tf.nn.bias_add(tf.matmul(l1, W2), b2, name="pred")

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        print "epoch %d" % epoch
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: [[x]], Y: [[y]]})

    graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ["pred"])

    with tf.gfile.GFile("output.pb", "wb") as f:
        f.write(graph_def.SerializeToString())
