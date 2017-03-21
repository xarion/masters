import tensorflow as tf
import numpy as np

import warnings

warnings.simplefilter("error")

INPUT_DIMENSIONS = 2
OUTPUT_DIMENSIONS = 1
BATCH_SIZE = 1000
HIDDEN_NODE_COUNT = 1000
SAMPLE_SIZE = 1000000
EPOCHS = 14
ACTIVATION_THRESHOLD = 0
# ACTIVATION_THRESHOLD = SAMPLE_SIZE * 0.08
LEARNING_RATE = 0.01

weights_not_initialized = True
pruned_node_count = 1

remaining_nodes = HIDDEN_NODE_COUNT
epochs = EPOCHS

while pruned_node_count > 0 and remaining_nodes > 1:
    with tf.Session() as session:
        input_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_DIMENSIONS])
        truth_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_DIMENSIONS])
        if weights_not_initialized:
            weights_1 = tf.Variable(tf.truncated_normal([INPUT_DIMENSIONS, HIDDEN_NODE_COUNT]))
            biases_1 = tf.Variable(tf.truncated_normal([HIDDEN_NODE_COUNT]))
            weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_NODE_COUNT, OUTPUT_DIMENSIONS]))
            biases_2 = tf.Variable(tf.truncated_normal([OUTPUT_DIMENSIONS]))
            weights_not_initialized = False
        else:
            weights_1 = tf.Variable(pruned_weights_1)
            biases_1 = tf.Variable(pruned_biases_1)
            weights_2 = tf.Variable(pruned_weights_2)
            biases_2 = tf.Variable(optimized_biases_2)

        hidden_nodes = tf.nn.relu(tf.matmul(input_placeholder, weights_1) + biases_1)
        output = tf.matmul(hidden_nodes, weights_2) + biases_2
        loss = tf.reduce_mean(tf.square(truth_placeholder - output))

        optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.3).minimize(loss)

        samples = np.random.random(size=[SAMPLE_SIZE, INPUT_DIMENSIONS])
        results = np.reshape(np.sum(samples, 1), [SAMPLE_SIZE, OUTPUT_DIMENSIONS])
        results += np.random.random(results.shape) * 0.0005

        session.run(tf.global_variables_initializer())

        for epoch in xrange(0, epochs):
            for batch_start in xrange(0, SAMPLE_SIZE, BATCH_SIZE):
                input_batch = samples[batch_start:batch_start + BATCH_SIZE, :]
                result_batch = results[batch_start:batch_start + BATCH_SIZE, :]

                feed_dict = {input_placeholder: input_batch, truth_placeholder: result_batch}
                batch_loss, _ = session.run([loss, optimizer], feed_dict=feed_dict)
            print "%d %f" % (epoch, batch_loss)

        print "trained for %d epoch(s), calculating loss with training data."

        node_activations = np.zeros([remaining_nodes])
        after_training_mean_loss = 0.

        for batch_start in xrange(0, SAMPLE_SIZE, BATCH_SIZE):
            input_batch = samples[batch_start:batch_start + BATCH_SIZE, :]
            result_batch = results[batch_start:batch_start + BATCH_SIZE, :]
            feed_dict = {input_placeholder: input_batch, truth_placeholder: result_batch}

            [hidden_node_values, batch_loss] = session.run([hidden_nodes, loss], feed_dict=feed_dict)

            after_training_mean_loss += batch_loss * BATCH_SIZE
            node_activations += np.sum(hidden_node_values > 0, 0)  # activated nodes are 1, rest 0

        after_training_mean_loss /= SAMPLE_SIZE
        print "Average loss for training data: %7f" % after_training_mean_loss

        kept_nodes = node_activations > ACTIVATION_THRESHOLD
        pruned_nodes = node_activations <= ACTIVATION_THRESHOLD
        pruned_node_count = np.sum(pruned_nodes)

        optimized_weights_1, optimized_biases_1, optimized_weights_2, optimized_biases_2 = session.run(
            [weights_1, biases_1, weights_2, biases_2])

        def distort(values):
            print "variance of values: %f" % np.var(values)
            values += (np.random.random(values.shape) * (1 - np.var(values))).astype(np.float32)
            return values

        pruned_weights_1 = distort(optimized_weights_1[:, kept_nodes])
        pruned_biases_1 = distort(optimized_biases_1[kept_nodes])

        pruned_weights_2 = distort(optimized_weights_2[kept_nodes, :])
        # pruned_biases_2 = distort(optimized_biases_2)  # there is no change in this.

        remaining_nodes -= pruned_node_count
        print "pruned: %d \t remaining: %d" % (pruned_node_count, remaining_nodes)
        epochs += 1
    session.close()
