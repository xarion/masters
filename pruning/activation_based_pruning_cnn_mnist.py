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

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set the seed to get stable results!
tf.set_random_seed(124)
np.random.seed(124)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

LEARNING_RATE = 0.01
EPOCHS = 2
BATCH_SIZE = 32
DISPLAY_STEP = 5
EXAMPLES_TO_SHOW = 10
PRUNING_ACTIVATION_THRESHOLD = 1200

NODE_COUNT_LAYER_1 = 32  # 1st layer num features // encoder
NODE_COUNT_LAYER_2 = 64  # 2nd layer num features // encoder
NODE_COUNT_LAYER_3 = 32  # 3rd layer num features // decoder
INPUT_CHANNELS = 1  # MNIST data input (img shape: 28*28) # now it's 1 because conv2d
KERNEL_SIZE = 3

node_count_layer_1 = NODE_COUNT_LAYER_1
node_count_layer_2 = NODE_COUNT_LAYER_2
node_count_layer_3 = NODE_COUNT_LAYER_3

node_count_histories = [[node_count_layer_1], [node_count_layer_2], [node_count_layer_3]]
training_steps = 0
validation_loss_history = []
step_durations = []


def batch_normalization(input_layer, parameters):
    return tf.nn.batch_normalization(input_layer,
                                     mean=parameters[0],
                                     variance=parameters[1],
                                     offset=parameters[2],
                                     scale=parameters[3],
                                     variance_epsilon=parameters[4])


def distort(values):
    values += (np.random.random(values.shape) * LEARNING_RATE).astype(np.float32)
    return values


def prune(weights, activation_counts, prune_axis):
    while np.mean(activation_counts) - 2 * np.std(activation_counts) < 0:
        minimum = np.min(activation_counts)
        weights = np.compress(activation_counts > minimum, weights, prune_axis)
        activation_counts = np.compress(activation_counts > minimum, activation_counts)
    activation_threshold = np.mean(activation_counts) - 2 * np.std(activation_counts)
    return np.compress(activation_counts > activation_threshold, weights, prune_axis)


weights_not_initialized = True
epochs = EPOCHS
cycles = 0
learning_rate = LEARNING_RATE

pruned_node_count = None
while pruned_node_count is None or pruned_node_count > 0:
    X = tf.placeholder("float", [BATCH_SIZE, 784])
    x_reshaped = tf.reshape(X, [BATCH_SIZE, 28, 28, 1])
    if weights_not_initialized:
        weights = {
            'encoder_1': tf.Variable(tf.random_normal([KERNEL_SIZE, KERNEL_SIZE, INPUT_CHANNELS, node_count_layer_1])),
            'encoder_2': tf.Variable(
                tf.random_normal([KERNEL_SIZE, KERNEL_SIZE, node_count_layer_1, node_count_layer_2])),
            'decoder_1': tf.Variable(
                tf.random_normal([KERNEL_SIZE, KERNEL_SIZE, node_count_layer_3, node_count_layer_2])),
            'decoder_2': tf.Variable(tf.random_normal([KERNEL_SIZE, KERNEL_SIZE, INPUT_CHANNELS, node_count_layer_3])),
        }
        biases = {
            'encoder_1': tf.Variable(tf.random_normal([node_count_layer_1])),
            'encoder_2': tf.Variable(tf.random_normal([node_count_layer_2])),
            'decoder_1': tf.Variable(tf.random_normal([node_count_layer_3])),
            'decoder_2': tf.Variable(tf.random_normal([INPUT_CHANNELS])),
        }
        batch_normalization_params = {
            'pre_processing': [tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                               tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                               tf.Variable(1e-5, dtype=tf.float32)],
            'encoder_1': [tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(1e-5, dtype=tf.float32)],
            'encoder_2': [tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(1e-5, dtype=tf.float32)],
            'decoder_1': [tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(1e-5, dtype=tf.float32)],
            'decoder_2': [tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(1e-5, dtype=tf.float32)],
        }
        weights_not_initialized = False
    else:
        weights = {
            'encoder_1': tf.Variable(pruned_weights_encoder_1),
            'encoder_2': tf.Variable(pruned_weights_encoder_2),
            'decoder_1': tf.Variable(pruned_weights_decoder_1),
            'decoder_2': tf.Variable(pruned_weights_decoder_2),
        }
        biases = {
            'encoder_1': tf.Variable(pruned_biases_encoder_1),
            'encoder_2': tf.Variable(pruned_biases_encoder_2),
            'decoder_1': tf.Variable(pruned_biases_decoder_1),
            'decoder_2': tf.Variable(biases_decoder_2),
        }
        batch_normalization_params = {}
        for key in updated_batch_normalization_params:
            batch_normalization_params[key] = [tf.Variable(val, tf.float32) for val in
                                               updated_batch_normalization_params[key]]

    with tf.name_scope("encoder_1"):
        encoder_1 = tf.add(tf.nn.conv2d(x_reshaped, weights['encoder_1'], strides=[1, 2, 2, 1], padding="SAME"),
                           biases['encoder_1'])
        encoder_1 = batch_normalization(encoder_1, batch_normalization_params["encoder_1"])
        encoder_1 = tf.nn.relu(encoder_1)

    with tf.name_scope("encoder_1"):
        encoder_2 = tf.add(tf.nn.conv2d(encoder_1, weights['encoder_2'], strides=[1, 2, 2, 1], padding="SAME"),
                           biases['encoder_2'])
        encoder_2 = batch_normalization(encoder_2, batch_normalization_params["encoder_2"])
        encoder_2 = tf.nn.relu(encoder_2)

    with tf.name_scope("decoder_1"):
        decoder_1 = tf.add(tf.nn.conv2d_transpose(encoder_2, weights['decoder_1'],
                                                  output_shape=[BATCH_SIZE, 14, 14, node_count_layer_3],
                                                  strides=[1, 2, 2, 1]), biases['decoder_1'])
        decoder_1 = batch_normalization(decoder_1, batch_normalization_params["decoder_1"])
        decoder_1 = tf.nn.relu(decoder_1)
    with tf.name_scope("decoder_2"):
        decoder_2 = tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(decoder_1, weights['decoder_2'],
                                                                output_shape=[BATCH_SIZE, 28, 28, INPUT_CHANNELS],
                                                                strides=[1, 2, 2, 1]), biases['decoder_2']))
        output = batch_normalization(decoder_2, batch_normalization_params["decoder_2"])

    with tf.name_scope("loss"):
        def l1_regularization():
            variables = list()
            variables.extend(biases.values())
            variables.extend(weights.values())
            return reduce(lambda k, l: k + l, map(lambda m: tf.reduce_sum(tf.abs(m)), variables))


        def activation_count_regularization():
            return reduce(lambda k, l: k + l,
                          map(lambda m:
                              tf.reduce_sum(tf.cast(m > 0, tf.float32)),
                              [encoder_1, encoder_2, decoder_1]))


        y_pred = tf.reshape(output, [BATCH_SIZE, 784])
        y_true = X
        # RMS + regularizers to encourage pruning
        cost = tf.reduce_mean(tf.pow((y_true - y_pred), 2))
        cost_regularizers = 0.000001 * (l1_regularization() + 0.001 * activation_count_regularization())
        regularized_cost = cost + cost_regularizers

    with tf.name_scope("optimizer"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(regularized_cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        total_batch = int(mnist.train.num_examples / BATCH_SIZE)
        training_start_time = datetime.now()
        for epoch in range(epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                _, c = sess.run([optimizer, regularized_cost], feed_dict={X: batch_xs})
                # print("cost=", "{:.9f}".format(c), "batch=%d/%d" % (i, total_batch))
            if epoch % DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(c))

        training_end_time = datetime.now()
        duration = (training_end_time - training_start_time).seconds
        step_durations.append(duration)
        print(step_durations)
        # Validation
        validation_cost = 0
        int(mnist.validation.num_examples / BATCH_SIZE)
        for i in range(total_batch):
            batch_xs, _ = mnist.validation.next_batch(BATCH_SIZE)
            batch_cost, = sess.run([cost], feed_dict={X: batch_xs})
            batch_cost *= (len(batch_xs) / mnist.validation.num_examples)
            validation_cost += batch_cost

        validation_loss_history.append(validation_cost)
        print(validation_loss_history)
        print("Training epocs finished, counting activations")

        ###
        # Here we go over the training dataset to determine which features are used
        # This note is to separate things from each other. It's getting too confusing otherwise.
        # @TODO: Separate things in functions so that it's more readable.
        ###
        encoder_1_activation_counts, encoder_2_activation_counts, decoder_1_activation_counts = (None, None, None)
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(BATCH_SIZE)

            batch_encoder_1_activation_counts, batch_encoder_2_activation_counts, batch_decoder_1_activation_counts = \
                sess.run(
                    [encoder_1, encoder_2, decoder_1],
                    feed_dict={X: batch_xs})
            if encoder_1_activation_counts is not None:
                encoder_1_activation_counts += np.sum(np.mean(batch_encoder_1_activation_counts, axis=(1, 2)), axis=0)
                encoder_2_activation_counts += np.sum(np.mean(batch_encoder_2_activation_counts, axis=(1, 2)), axis=0)
                decoder_1_activation_counts += np.sum(np.mean(batch_decoder_1_activation_counts, axis=(1, 2)), axis=0)
            else:
                encoder_1_activation_counts = np.sum(np.mean(batch_encoder_1_activation_counts, axis=(1, 2)), axis=0)
                encoder_2_activation_counts = np.sum(np.mean(batch_encoder_2_activation_counts, axis=(1, 2)), axis=0)
                decoder_1_activation_counts = np.sum(np.mean(batch_decoder_1_activation_counts, axis=(1, 2)), axis=0)

        weights_encoder_1, weights_encoder_2, biases_encoder_1, biases_encoder_2 = sess.run(
            [weights['encoder_1'], weights['encoder_2'], biases['encoder_1'], biases['encoder_2']])

        pruned_weights_encoder_1 = prune(weights_encoder_1, encoder_1_activation_counts, prune_axis=3)
        pruned_biases_encoder_1 = prune(biases_encoder_1, encoder_1_activation_counts, prune_axis=0)
        pruned_weights_encoder_2 = prune(weights_encoder_2, encoder_2_activation_counts, prune_axis=3)
        pruned_weights_encoder_2 = prune(pruned_weights_encoder_2, encoder_1_activation_counts, prune_axis=2)
        pruned_biases_encoder_2 = prune(biases_encoder_2, encoder_2_activation_counts, prune_axis=0)

        weights_decoder_1, weights_decoder_2, biases_decoder_1, biases_decoder_2 = sess.run(
            [weights['decoder_1'], weights['decoder_2'], biases['decoder_1'], biases['decoder_2']])

        pruned_weights_decoder_1 = prune(weights_decoder_1, encoder_2_activation_counts, prune_axis=3)
        pruned_weights_decoder_1 = prune(pruned_weights_decoder_1, decoder_1_activation_counts, prune_axis=2)
        pruned_weights_decoder_2 = prune(weights_decoder_2, decoder_1_activation_counts, prune_axis=3)
        pruned_biases_decoder_1 = prune(biases_decoder_1, decoder_1_activation_counts, prune_axis=0)

        print("encoder 1, shape before: %s, shape_after: %s" % (
            str(weights_encoder_1.shape), str(pruned_weights_encoder_1.shape)))
        print("encoder 2, shape before: %s, shape_after: %s" % (
            str(weights_encoder_2.shape), str(pruned_weights_encoder_2.shape)))
        print("decoder 1, shape before: %s, shape_after: %s" % (
            str(weights_decoder_1.shape), str(pruned_weights_decoder_1.shape)))
        print("decoder 2, shape before: %s, shape_after: %s" % (
            str(weights_decoder_2.shape), str(pruned_weights_decoder_2.shape)))

        pruned_node_count = 0

        pruned_node_count += node_count_layer_1 - pruned_weights_encoder_1.shape[3]
        node_count_layer_1 = pruned_weights_encoder_1.shape[3]

        pruned_node_count += node_count_layer_2 - pruned_weights_encoder_2.shape[3]
        node_count_layer_2 = pruned_weights_encoder_2.shape[3]

        pruned_node_count += node_count_layer_3 - pruned_weights_decoder_1.shape[2]
        node_count_layer_3 = pruned_weights_decoder_1.shape[2]

        node_count_histories[0].append(node_count_layer_1)
        node_count_histories[1].append(node_count_layer_2)
        node_count_histories[2].append(node_count_layer_3)
        print(node_count_histories)
        print("Total number of pruned nodes: %d" % (pruned_node_count))

        pruned_weights_encoder_1 = distort(pruned_weights_encoder_1)
        pruned_weights_encoder_2 = distort(pruned_weights_encoder_2)
        pruned_weights_decoder_1 = distort(pruned_weights_decoder_1)
        pruned_weights_decoder_2 = distort(pruned_weights_decoder_2)

        updated_batch_normalization_params = {}
        for key in batch_normalization_params:
            updated_batch_normalization_params[key] = [val.eval() for val in batch_normalization_params[key]]

        batch_normalization_params = {
            'pre_processing': [tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                               tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                               tf.Variable(1e-5, dtype=tf.float32)],
            'encoder_1': [tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(1e-5, dtype=tf.float32)],
            'encoder_2': [tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(1e-5, dtype=tf.float32)],
            'decoder_1': [tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(1e-5, dtype=tf.float32)],
            'decoder_2': [tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(0, dtype=tf.float32), tf.Variable(1, dtype=tf.float32),
                          tf.Variable(1e-5, dtype=tf.float32)],
        }

        if pruned_node_count == 0:
            # Test
            test_cost = .0
            int(mnist.test.num_examples / BATCH_SIZE)
            for i in range(total_batch):
                batch_xs, _ = mnist.test.next_batch(BATCH_SIZE)
                batch_cost, = sess.run([cost], feed_dict={X: batch_xs})
                batch_cost *= (len(batch_xs) / mnist.test.num_examples)
                test_cost += batch_cost

            print("final cost in the test dataset: %.9f" % test_cost)
            encode_decode = sess.run(
                y_pred, feed_dict={X: mnist.test.images[:BATCH_SIZE]})

            plt.figure()
            f, a = plt.subplots(2, 10, figsize=(10, 2))
            for i in range(EXAMPLES_TO_SHOW):
                a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
                a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
            f.show()
            plt.draw()

plt.figure()
num_cycles = len(node_count_histories[0])
layer_1, = plt.plot(range(0, num_cycles), node_count_histories[0], label='Layer 1')
layer_2, = plt.plot(range(0, num_cycles), node_count_histories[1], label='Layer 2')
layer_3, = plt.plot(range(0, num_cycles), node_count_histories[2], label='Layer 3')
plt.legend(handles=[layer_1, layer_2, layer_3])
plt.title('Number of features per layer over time')
plt.xlabel('Training Cycle')
plt.ylabel('Remaining Features')

plt.figure()
plt.plot(range(1, num_cycles), validation_loss_history)
plt.title('Validation Loss over Cycles')
plt.xlabel('Training Cycle')
plt.ylabel('Loss')

plt.figure()
plt.plot(range(1, num_cycles), step_durations)
plt.title('Duration of Training Cycles over time')
plt.xlabel('Training Cycle')
plt.ylabel('Seconds')
plt.show()
