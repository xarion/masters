import warnings

import numpy as np
import tensorflow as tf

warnings.simplefilter("error")

INPUT_DIMENSIONS = 2
OUTPUT_DIMENSIONS = 1
BATCH_SIZE = 1000
HIDDEN_NODE_COUNT = 1000
SAMPLE_SIZE = 1000000
EPOCHS = 3
ACTIVATION_THRESHOLD = 0
# ACTIVATION_THRESHOLD = SAMPLE_SIZE * 0.08
LEARNING_RATE = 0.01

weights_not_initialized = True
pruned_node_count = 1
remaining_nodes = HIDDEN_NODE_COUNT
epochs = EPOCHS

seed_base = 126
np.random.seed(seed=seed_base + seed_base + 1)
samples = np.random.random(size=[SAMPLE_SIZE, INPUT_DIMENSIONS])
results = np.reshape(np.sum(samples, 1), [SAMPLE_SIZE, OUTPUT_DIMENSIONS])
# results += np.random.random(results.shape) * 0.0005

while pruned_node_count > 0 and remaining_nodes > 1:
    with tf.Session() as session:
        tf.set_random_seed(seed_base + 2)
        input_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_DIMENSIONS])
        truth_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_DIMENSIONS])
        if weights_not_initialized:
            weights_1 = tf.Variable(tf.truncated_normal([INPUT_DIMENSIONS, HIDDEN_NODE_COUNT], seed=seed_base + 3))
            biases_1 = tf.Variable(tf.truncated_normal([HIDDEN_NODE_COUNT], seed=seed_base + 4))
            weights_2 = tf.Variable(tf.truncated_normal([HIDDEN_NODE_COUNT, OUTPUT_DIMENSIONS], seed=seed_base + 5))
            biases_2 = tf.Variable(tf.truncated_normal([OUTPUT_DIMENSIONS], seed=seed_base + 6))
            weights_not_initialized = False
        else:
            weights_1 = tf.Variable(pruned_weights_1)
            biases_1 = tf.Variable(pruned_biases_1)
            weights_2 = tf.Variable(pruned_weights_2)
            biases_2 = tf.Variable(optimized_biases_2)

        hidden_nodes = tf.nn.relu(tf.matmul(input_placeholder, weights_1) + biases_1)
        output = tf.matmul(hidden_nodes, weights_2) + biases_2
        loss = tf.reduce_mean(tf.square(truth_placeholder - output))

        l1_regularizer = tf.reduce_mean(tf.concat(map(lambda x: tf.reshape(tf.abs(x), [-1]),
                                                      [weights_1, biases_1, weights_2, biases_2]), 0))

        cost = loss # + (l1_regularizer / tf.square(float(remaining_nodes)))

        optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, momentum=0.3)
        gradients = optimizer.compute_gradients(cost, var_list=[weights_1, biases_1, weights_2, biases_2])
        optimize = optimizer.minimize(cost)

        session.run(tf.global_variables_initializer())
        for epoch in xrange(0, epochs):
            mean_grads_w1 = np.zeros([2, remaining_nodes])
            mean_grads_b1 = np.zeros([remaining_nodes])
            mean_grads_w2 = np.zeros([remaining_nodes, 1])
            mean_grads_b2 = np.zeros([1])
            for batch_start in xrange(0, SAMPLE_SIZE, BATCH_SIZE):
                input_batch = samples[batch_start:batch_start + BATCH_SIZE, :]
                result_batch = results[batch_start:batch_start + BATCH_SIZE, :]

                feed_dict = {input_placeholder: input_batch, truth_placeholder: result_batch}
                batch_loss, batch_cost, batch_l1_regularizer, grads, w1, _ = session.run(
                    [loss, cost, l1_regularizer, gradients, weights_1, optimize], feed_dict=feed_dict)
                mean_grads_w1 += grads[0][0]
                mean_grads_b1 += grads[1][0]
                mean_grads_w2 += grads[2][0]
                mean_grads_b2 += grads[3][0]
                # if remaining_nodes < 1000 and batch_start % 10000 == 0:
                #     print "batch_start: " + str(batch_start)
                #     print "Grads: "
                #     print grads
                #     print "Weight_1"
                #     print w1
            mean_grads_w1 /= (SAMPLE_SIZE / BATCH_SIZE)
            print "weights 1"
            print "max: %.5f min : %.5f" % (mean_grads_w1.max(), mean_grads_w1.min())
            print "std: %.5f mean: %.5f" % (mean_grads_w1.std(), mean_grads_w1.mean())
            print "biases 1"
            print "max: %.5f min : %.5f" % (mean_grads_b1.max(), mean_grads_b1.min())
            print "std: %.5f mean: %.5f" % (mean_grads_b1.std(), mean_grads_b1.mean())
            print "weights 2"
            print "max: %.5f min : %.5f" % (mean_grads_w2.max(), mean_grads_w2.min())
            print "std: %.5f mean: %.5f" % (mean_grads_w2.std(), mean_grads_w2.mean())
            print "biases 2"
            print "max: %.5f min : %.5f" % (mean_grads_b2.max(), mean_grads_b2.min())
            print "std: %.5f mean: %.5f" % (mean_grads_b2.std(), mean_grads_b2.mean())
            print "epoch: %d batch_cost: %f batch_loss: %f batch_l1_regularizer: %f" % (
                epoch, batch_cost, batch_loss, batch_l1_regularizer)

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


        def distort(values):
            # print "variance of values: %f" % np.var(values)
            values += (np.random.random(values.shape) * (1 - np.var(values))).astype(np.float32)
            return values


        optimized_weights_1, optimized_biases_1, optimized_weights_2, optimized_biases_2 = session.run(
            [weights_1, biases_1, weights_2, biases_2])

        if pruned_node_count != remaining_nodes:
            pruned_weights_1 = distort(optimized_weights_1[:, kept_nodes])
            pruned_biases_1 = distort(optimized_biases_1[kept_nodes])
            pruned_weights_2 = distort(optimized_weights_2[kept_nodes, :])
            # pruned_biases_2 = distort(optimized_biases_2)  # there is no change in this.
            remaining_nodes -= pruned_node_count
            print "pruned: %d \t remaining: %d" % (pruned_node_count, remaining_nodes)
        else:
            pruned_weights_1 = distort(optimized_weights_1)
            pruned_biases_1 = distort(optimized_biases_1)
            pruned_weights_2 = distort(optimized_weights_2)
        epochs += 1
    session.close()

print "w1: "
print pruned_weights_1
print "b1: "
print pruned_biases_1
print "w2: "
print pruned_weights_2
print "b2: "
print pruned_biases_1
