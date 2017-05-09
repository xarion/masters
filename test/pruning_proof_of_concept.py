from __future__ import print_function
from __future__ import print_function

import unittest

import numpy as np
import tensorflow as tf

CHECKPOINT_FOLDER = "test_checkpoint_folder"
CHECKPOINT = CHECKPOINT_FOLDER + "/test_checkpoint"


class LoadPrunedModel(unittest.TestCase):
    def test_load_pruned_model(self):
        def get_model():
            with tf.variable_scope("foo"):
                v = tf.get_variable("bar", [3, 3, 5, 11], initializer=tf.ones_initializer(), validate_shape=False)
                conv = tf.nn.conv2d(tf.ones([1, 5, 5, 5]), v, strides=[1, 1, 1, 1], padding="SAME")
            return v, conv

        def get_modified_model():
            with tf.variable_scope("foo"):
                v = tf.get_variable("bar", [3, 3, 5, 5], initializer=tf.zeros_initializer(), validate_shape=False)
                conv = tf.nn.conv2d(tf.ones([1, 5, 5, 5]), v, strides=[1, 1, 1, 1], padding="SAME")
            return v, conv

        #  save a checkpoint file for the modified model which has a "pruned" filter
        g = tf.Graph()
        with tf.Session(graph=g) as session:
            weight_tensor, op = get_modified_model()
            saver = tf.train.Saver()
            session.run(tf.global_variables_initializer())

            shape, weights, = session.run([tf.shape(op), weight_tensor])
            np.testing.assert_array_equal(shape, [1, 5, 5, 5])
            np.testing.assert_array_equal(weights, np.zeros([3, 3, 5, 5]))

            saver.save(session, CHECKPOINT)
            session.close()

        # load the checkpoint file for a model with different size
        starting_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
        assert starting_checkpoint is not None

        #  define the starting network but load the pruned kernels with different shapes/values.
        g = tf.Graph()
        with tf.Session(graph=g) as session:
            weight_tensor, op = get_model()
            saver = tf.train.Saver()
            saver.restore(session, starting_checkpoint)

            shape, weights, = session.run([tf.shape(op), weight_tensor])

            np.testing.assert_array_equal(shape, [1, 5, 5, 5])
            np.testing.assert_array_equal(weights, np.zeros([3, 3, 5, 5]))
            session.close()

    def test_assign_and_load_pruned_weights(self):
        def get_model():
            with tf.variable_scope("foo"):
                v = tf.get_variable("bar", [3, 3, 5, 11], initializer=tf.ones_initializer(), validate_shape=False)
                conv = tf.nn.conv2d(tf.ones([1, 5, 5, 5]), v, strides=[1, 1, 1, 1], padding="SAME")
            return v, conv

        def prune(weight_tensor):
            return tf.assign(weight_tensor, tf.zeros([3, 3, 5, 5]), validate_shape=False)

        g = tf.Graph()
        with tf.Session(graph=g) as session:
            weight_tensor, op = get_model()
            prune_op = prune(weight_tensor)
            saver = tf.train.Saver()
            session.run(tf.global_variables_initializer())

            shape, weights, = session.run([tf.shape(op), weight_tensor])
            np.testing.assert_array_equal(shape, [1, 5, 5, 11])
            np.testing.assert_array_equal(weights, np.ones([3, 3, 5, 11]))

            _, weights = session.run([prune_op, weight_tensor])
            weights, = session.run([weight_tensor])

            np.testing.assert_array_equal(weights, np.zeros([3, 3, 5, 5]))

            saver.save(session, CHECKPOINT)
            session.close()

        # load the checkpoint file for a model with different size
        starting_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
        assert starting_checkpoint is not None

        #  define the starting network but load the pruned kernels with different shapes/values.
        g = tf.Graph()
        with tf.Session(graph=g) as session:
            weight_tensor, op = get_model()
            saver = tf.train.Saver()
            saver.restore(session, starting_checkpoint)

            shape, weights, = session.run([tf.shape(op), weight_tensor])

            np.testing.assert_array_equal(shape, [1, 5, 5, 5])
            np.testing.assert_array_equal(weights, np.zeros([3, 3, 5, 5]))
            session.close()


if __name__ == '__main__':
    unittest.main()
