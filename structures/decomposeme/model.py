from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

tf.set_random_seed(245)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(input_layer, filter_size, in_channels, out_channels, intermediate_channels, valid_padding=False):
    strides = [1, 1, 1, 1]
    padding = 'SAME' if not valid_padding else 'VALID'
    W_1 = weight_variable([filter_size, 1, in_channels, intermediate_channels])
    b_1 = bias_variable([intermediate_channels])
    W_2 = weight_variable([1, filter_size, intermediate_channels, out_channels])
    b_2 = bias_variable([out_channels])
    first = tf.nn.relu(tf.nn.conv2d(input_layer, W_1, strides, padding) + b_1)
    return tf.nn.relu(tf.nn.conv2d(first, W_2, strides, padding) + b_2)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = conv2d(x_image, filter_size=5, in_channels=1, out_channels=32, intermediate_channels=12)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = conv2d(h_pool1, filter_size=5, in_channels=32, out_channels=64, intermediate_channels=48)
h_pool2 = max_pool_2x2(h_conv2)

h_fc1 = conv2d(h_pool2, filter_size=7, in_channels=64, out_channels=1024, intermediate_channels=256, valid_padding=True)
h_fc1 = tf.reshape(h_fc1, [-1, 1024])
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(201):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels}))
