import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util

learning_rate = 0.01
training_epochs = 100
display_step = 50
hidden_nodes = 124
# Training Data
train_X = np.random.beta(0.5, 5, size=[1000, 1])
train_Y = np.random.beta(0.1, 1, size=[1000, 1])
train_X.sort(axis=0)
train_Y.sort(axis=0)

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="input")
Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

W1 = tf.Variable(tf.random_normal((1, hidden_nodes)), name="l1_weights", dtype=tf.float32)
b1 = tf.Variable(tf.random_normal([hidden_nodes]), name="l1_bias", dtype=tf.float32)

W2 = tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes]), name="l2_weights", dtype=tf.float32)
b2 = tf.Variable(tf.random_normal([hidden_nodes]), name="l2_bias", dtype=tf.float32)

W3 = tf.Variable(tf.random_normal([hidden_nodes, 1]), name="out_weights", dtype=tf.float32)
b3 = tf.Variable(tf.random_normal([1]), name="out_bias", dtype=tf.float32)

l1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(X, W1), b1, name="l1"))
l2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l1, W2), b2, name="l2"))
pred = tf.nn.bias_add(tf.matmul(l2, W3), b3, name="pred")

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples) + \
       0.0005 * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))
# Gradient descent
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    _cost = 100
    # Fit all training data
    while _cost > 1:
        _opt, _cost = sess.run([optimizer, cost], feed_dict={X: train_X, Y: train_Y})
        print _cost

    graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ["pred"])

    with tf.gfile.GFile("output.pb", "wb") as f:
        f.write(graph_def.SerializeToString())
