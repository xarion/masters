import unittest

import numpy as np
import tensorflow as tf

from model.pruning import OpPruning


class TestPruningByIndex(unittest.TestCase):
    def test_pruning_with_dynamic_partition(self):
        with tf.Session() as sess:
            weight_shape = [3, 3, 32, 64]
            initial_weights = np.random.random(weight_shape)
            weights = tf.Variable(initial_weights, dtype=tf.float32)
            weights.initializer.run()
            input_indices = np.unique(np.random.randint(0, weight_shape[2], int(weight_shape[2] / 2))).astype(np.int32)

            slices = np.zeros(weight_shape)
            slices[:, :, input_indices, :] = 1

            partition = tf.dynamic_partition(weights, slices, num_partitions=2)[1]
            partition_reshape = tf.reshape(partition,
                                           (weight_shape[0], weight_shape[1], len(input_indices), weight_shape[3]))
            prune = tf.assign(weights, partition_reshape, validate_shape=False)

            sess.run([prune])
            pruned_weights, = sess.run([weights])
            numpy_pruned_weights = initial_weights[:, :, input_indices, :]
            np.testing.assert_array_almost_equal(numpy_pruned_weights, pruned_weights)

    def test_pruning_function(self):
        with tf.Session() as sess:
            weight_shape = [3, 3, 32, 64]
            initial_weights = np.random.random(weight_shape)
            weight_tensor = tf.Variable(initial_weights, dtype=tf.float32)
            keep_indices = np.unique(np.random.randint(0, weight_shape[2], int(weight_shape[2] / 2))).astype(np.int32)
            pruner = OpPruning()
            prune = pruner.prune(weight_tensor, keep_indices, 2)

            sess.run([tf.variables_initializer(tf.global_variables())])
            weights, = sess.run([weight_tensor])

            np.testing.assert_array_almost_equal(weights, initial_weights)

            sess.run([prune])
            weights, = sess.run([weight_tensor])
            np.testing.assert_array_almost_equal(weights, initial_weights[:, :, keep_indices, :])


if __name__ == '__main__':
    unittest.main()
