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

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# set the seed to get stable results!
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('debugging', True,
                            """Whether to show detailed log messages.""")

tf.app.flags.DEFINE_boolean('prune_on_activation_count', False,
                            """Whether to prune with activation counts or activation values.""")

tf.app.flags.DEFINE_boolean('distort_weights', True,
                            """Whether to distort weights after pruning""")

tf.app.flags.DEFINE_boolean('distort_by_kernel', False,
                            """Whether to apply distort each N by N kernel with one value, 
                            or apply separate distortions to every value""")

tf.app.flags.DEFINE_boolean('prune_outliers', True,
                            """Whether to prune the outliers or by activation threshold""")

tf.app.flags.DEFINE_boolean('l1_regularization', False,
                            """Whether to apply L1 Regularization or not""")

tf.app.flags.DEFINE_boolean('activation_count_regularization', False,
                            """Whether to apply "activation count regularization" or not""")

tf.app.flags.DEFINE_boolean('increment_epochs_after_traning_cycle', False,
                            """Whether to prune the outliers or by activation threshold""")

tf.app.flags.DEFINE_boolean('plot_figures', True,
                            """Whether to prune the outliers or by activation threshold""")

tf.app.flags.DEFINE_integer('activation_threshold', 0,
                            """Threshold to be used while determining irrelevant hidden neurons.""")

tf.app.flags.DEFINE_integer('std_multiplier', 2,
                            """Multiplier of standard deviation while determining the outliers.""")

tf.app.flags.DEFINE_integer('initial_epochs', 1,
                            """Number of Epochs per training cycle""")

tf.app.flags.DEFINE_integer('seed', 1234,
                            """random number seed""")

tf.app.flags.DEFINE_float('l1_regularization_multiplier', 0.001,
                          """the multiplier for l1_regularizer for the cost""")

tf.app.flags.DEFINE_float('activation_count_multiplier', 0.001,
                          """the multiplier for activation count regularizer for the cost""")

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

print(
    """--debugging %s --prune_on_activation_count %s --distort_weights %s --distort_by_kernel %s --prune_outliers %s --std_multiplier %s --activation_threshold %s --l1_regularization %s --l1_regularization_multiplier %s --activation_count_regularization %s --activation_count_multiplier %s --increment_epochs_after_traning_cycle %s --plot_figures %s --initial_epochs %s --seed %s""" % (
        FLAGS.debugging, FLAGS.prune_on_activation_count, FLAGS.distort_weights, FLAGS.distort_by_kernel,
        FLAGS.prune_outliers, FLAGS.std_multiplier, FLAGS.activation_threshold, FLAGS.l1_regularization,
        FLAGS.l1_regularization_multiplier, FLAGS.activation_count_regularization, FLAGS.activation_count_multiplier,
        FLAGS.increment_epochs_after_traning_cycle, FLAGS.plot_figures, FLAGS.initial_epochs, FLAGS.seed))


def main(FLAGS):
    DEBUG = FLAGS.debugging

    SEED = FLAGS.seed
    np.random.seed(SEED)

    LEARNING_RATE = 0.005
    EPOCHS = FLAGS.initial_epochs
    BATCH_SIZE = 32
    DISPLAY_STEP = 10
    EXAMPLES_TO_SHOW = 10

    NODE_COUNT_LAYER_1 = 32  # 1st layer num features // encoder
    NODE_COUNT_LAYER_2 = 64  # 2nd layer num features // encoder
    NODE_COUNT_LAYER_3 = 32  # 3rd layer num features // decoder
    INPUT_CHANNELS = 1  # MNIST data input (img shape: 28*28) # now it's 1 because conv2d
    KERNEL_SIZE = 3

    node_count_layer_1 = NODE_COUNT_LAYER_1
    node_count_layer_2 = NODE_COUNT_LAYER_2
    node_count_layer_3 = NODE_COUNT_LAYER_3

    node_count_histories = [[node_count_layer_1], [node_count_layer_2], [node_count_layer_3]]
    training_cycles = 0
    validation_loss_history = []
    step_durations = []

    def batch_normalization(input_layer, parameters):
        return input_layer

    def distort(values):
        if FLAGS.distort_weights:
            shape = values.shape

            if FLAGS.distort_by_kernel:
                shape = (1, 1, shape[2], shape[3])

            values += (np.random.random(shape) * LEARNING_RATE).astype(np.float32)
        return values

    def get_counts(values):
        if FLAGS.prune_on_activation_count:
            values = (values > 0).astype(int)
        return np.sum(np.mean(values, axis=(1, 2)), axis=0)

    def prune(weights, activation_counts, prune_axis):
        if FLAGS.prune_outliers:
            while np.mean(activation_counts) - FLAGS.std_multiplier * np.std(activation_counts) < 0:
                minimum = np.min(activation_counts)
                weights = np.compress(activation_counts > minimum, weights, prune_axis)
                activation_counts = np.compress(activation_counts > minimum, activation_counts)
            activation_threshold = np.mean(activation_counts) - FLAGS.std_multiplier * np.std(activation_counts)
        else:
            activation_threshold = FLAGS.activation_threshold
        return np.compress(activation_counts > activation_threshold, weights, prune_axis)

    weights_not_initialized = True
    epochs = EPOCHS

    pruned_node_count = None
    while (pruned_node_count is None or pruned_node_count > 0) and\
            (node_count_layer_1 > 0 and node_count_layer_2 > 0 and node_count_layer_3 > 0):
        X = tf.placeholder("float", [BATCH_SIZE, 784])
        x_reshaped = tf.reshape(X, [BATCH_SIZE, 28, 28, 1])
        if weights_not_initialized:
            weights = {
                'encoder_1': tf.Variable(
                    tf.random_normal([KERNEL_SIZE, KERNEL_SIZE, INPUT_CHANNELS, node_count_layer_1], seed=SEED + 1)),
                'encoder_2': tf.Variable(
                    tf.random_normal([KERNEL_SIZE, KERNEL_SIZE, node_count_layer_1, node_count_layer_2],
                                     seed=SEED + 2)),
                'decoder_1': tf.Variable(
                    tf.random_normal([KERNEL_SIZE, KERNEL_SIZE, node_count_layer_3, node_count_layer_2],
                                     seed=SEED + 3)),
                'decoder_2': tf.Variable(
                    tf.random_normal([KERNEL_SIZE, KERNEL_SIZE, INPUT_CHANNELS, node_count_layer_3], seed=SEED + 4)),
            }
            biases = {
                'encoder_1': tf.Variable(tf.zeros([node_count_layer_1]), trainable=False),
                'encoder_2': tf.Variable(tf.zeros([node_count_layer_2]), trainable=False),
                'decoder_1': tf.Variable(tf.zeros([node_count_layer_3]), trainable=False),
                'decoder_2': tf.Variable(tf.zeros([INPUT_CHANNELS]), trainable=False),
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
                'encoder_1': tf.Variable(pruned_biases_encoder_1, trainable=False),
                'encoder_2': tf.Variable(pruned_biases_encoder_2, trainable=False),
                'decoder_1': tf.Variable(pruned_biases_decoder_1, trainable=False),
                'decoder_2': tf.Variable(biases_decoder_2, trainable=False),
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
            decoder_2 = tf.nn.tanh(tf.add(tf.nn.conv2d_transpose(decoder_1, weights['decoder_2'],
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
            cost = tf.reduce_mean(tf.pow((y_true - y_pred), 2))

            cost_regularizers = 0
            if FLAGS.l1_regularization:
                cost_regularizers += FLAGS.l1_regularization_multiplier * l1_regularization()
            if FLAGS.activation_count_regularization:
                cost_regularizers += FLAGS.activation_count_multiplier * activation_count_regularization()

            regularized_cost = cost + cost_regularizers

        with tf.name_scope("optimizer"):
            step = tf.Variable(0)
            boundaries = [600, 1100, 1400]
            values = [0.001, 0.0001, 0.00001, 0.000001]
            learning_rate = tf.train.piecewise_constant(step, boundaries, values)

            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(regularized_cost,
                                                                                                      global_step=step)
            # optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(regularized_cost)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            tf.set_random_seed(SEED)
            sess.run(init)

            total_batch = int(mnist.train.num_examples / BATCH_SIZE)
            training_start_time = datetime.now()
            for epoch in range(epochs):
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                    _, c = sess.run([optimizer, regularized_cost], feed_dict={X: batch_xs})
                    if i % DISPLAY_STEP == 0:
                        print("cost=", "{:.9f}".format(c), "batch=%d/%d" % (i, total_batch)) if DEBUG else None

            training_end_time = datetime.now()
            duration = (training_end_time - training_start_time).seconds
            step_durations.append(duration)
            print(step_durations) if DEBUG else None
            # Validation
            validation_cost = 0
            int(mnist.validation.num_examples / BATCH_SIZE)
            for i in range(total_batch):
                batch_xs, _ = mnist.validation.next_batch(BATCH_SIZE)
                batch_cost, = sess.run([cost], feed_dict={X: batch_xs})
                batch_cost *= (len(batch_xs) / mnist.validation.num_examples)
                validation_cost += batch_cost

            validation_loss_history.append(validation_cost)
            print(validation_loss_history) if DEBUG else None
            print("Training epocs finished, counting activations") if DEBUG else None

            ###
            # Here we go over the training dataset to determine which features are used
            # This note is to separate things from each other. It's getting too confusing otherwise.
            # @TODO: Separate things in functions so that it's more readable.
            ###
            encoder_1_activation_counts, encoder_2_activation_counts, decoder_1_activation_counts = (None, None, None)

            for i in range(total_batch):
                batch_xs, _ = mnist.train.next_batch(BATCH_SIZE)

                batch_encoder_1_activations, batch_encoder_2_activations, batch_decoder_1_activations = \
                    sess.run(
                        [encoder_1, encoder_2, decoder_1],
                        feed_dict={X: batch_xs})
                if encoder_1_activation_counts is not None:
                    encoder_1_activation_counts += get_counts(batch_encoder_1_activations)
                    encoder_2_activation_counts += get_counts(batch_encoder_2_activations)
                    decoder_1_activation_counts += get_counts(batch_decoder_1_activations)
                else:
                    encoder_1_activation_counts = get_counts(batch_encoder_1_activations)
                    encoder_2_activation_counts = get_counts(batch_encoder_2_activations)
                    decoder_1_activation_counts = get_counts(batch_decoder_1_activations)

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
                str(weights_encoder_1.shape), str(pruned_weights_encoder_1.shape))) if DEBUG else None
            print("encoder 2, shape before: %s, shape_after: %s" % (
                str(weights_encoder_2.shape), str(pruned_weights_encoder_2.shape))) if DEBUG else None
            print("decoder 1, shape before: %s, shape_after: %s" % (
                str(weights_decoder_1.shape), str(pruned_weights_decoder_1.shape))) if DEBUG else None
            print("decoder 2, shape before: %s, shape_after: %s" % (
                str(weights_decoder_2.shape), str(pruned_weights_decoder_2.shape))) if DEBUG else None

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
            print(node_count_histories) if DEBUG else None
            print("Total number of pruned nodes: %d" % (pruned_node_count)) if DEBUG else None

            pruned_weights_encoder_1 = distort(pruned_weights_encoder_1)
            pruned_weights_encoder_2 = distort(pruned_weights_encoder_2)
            pruned_weights_decoder_1 = distort(pruned_weights_decoder_1)
            pruned_weights_decoder_2 = distort(pruned_weights_decoder_2)

            updated_batch_normalization_params = {}
            for key in batch_normalization_params:
                updated_batch_normalization_params[key] = [val.eval() for val in batch_normalization_params[key]]

            if pruned_node_count == 0:
                # Test
                test_cost = .0
                int(mnist.test.num_examples / BATCH_SIZE)
                for i in range(total_batch):
                    batch_xs, _ = mnist.test.next_batch(BATCH_SIZE)
                    batch_cost, = sess.run([cost], feed_dict={X: batch_xs})
                    batch_cost *= (len(batch_xs) / mnist.test.num_examples)
                    test_cost += batch_cost

                print("final cost in the test dataset: %.9f" % test_cost) if DEBUG else None
                encode_decode = sess.run(
                    y_pred, feed_dict={X: mnist.test.images[:BATCH_SIZE]})
                if FLAGS.plot_figures:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    f, a = plt.subplots(2, 10, figsize=(10, 2))
                    for i in range(EXAMPLES_TO_SHOW):
                        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
                        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
                    f.show()
                    plt.draw()
        sess.close()
        if FLAGS.increment_epochs_after_traning_cycle:
            epochs += 1

    print("Node Count Histories: %s" % str(node_count_histories))
    print("validation losses in time: %s" % str(validation_loss_history))
    print("Step Durations:  %s" % str(step_durations))
    print("Total Training Cycles: %d" % training_cycles)
    if FLAGS.plot_figures:
        import matplotlib.pyplot as plt
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


main(FLAGS)
