import unittest

import numpy as np
import tensorflow as tf

CHECKPOINT_FOLDER = "test_checkpoint_folder"
CHECKPOINT = CHECKPOINT_FOLDER + "/test_checkpoint"


class LoadPrunedModelAdam(unittest.TestCase):
    def test_load_pruned_weights_with_adam_optimizer(self):

        #  if we store the pruned shapes, we can potentially use this information
        #  when we're recreating these variables in the future.
        known_variable_shapes = {}

        def get_model():
            def custom_getter(getter, name, shape, *args, **kwargs):
                full_name = "%s:0" % (name)
                if full_name in known_variable_shapes:
                    shape = known_variable_shapes[full_name]
                else:
                    known_variable_shapes[full_name] = shape
                return getter(name, shape, *args, **kwargs)

            with tf.variable_scope("foo"):
                conv_filters = tf.get_variable("filters", [3, 3, 5, 11],
                                               initializer=tf.ones_initializer(),
                                               custom_getter=custom_getter)
                conv_op = tf.nn.conv2d(tf.ones([1, 5, 5, 5]), conv_filters, strides=[1, 1, 1, 1], padding="SAME")

            return conv_filters, conv_op

        def get_training_step(output_op):
            with tf.variable_scope("train"):
                loss = tf.reshape(tf.reduce_mean(output_op - 0.1), [1])
                train_step = tf.train.AdamOptimizer().minimize(loss)
            return train_step

        def prune(tensor):
            tensor_name = tensor.name
            adam_tensor_name_part = "/" + tensor_name.split(':')[0] + '/Adam'
            ops = list()
            for other_tensor in tf.all_variables():
                if other_tensor.name.find(adam_tensor_name_part) >= 0:
                    if other_tensor.get_shape() == tensor.get_shape():
                        ops.append(prune_tensor_op(other_tensor))
            prune_op = prune_tensor_op(tensor)
            ops.append(prune_op)
            known_variable_shapes[tensor_name] = prune_op.get_shape()
            return ops

        #  this is a mock pruning operation, in reality this will be using a method like tf.gather to prune variables
        #  in theory, we can prune the moment variables of Adam in same indices.
        def prune_tensor_op(tensor):
            return tf.assign(tensor, tf.zeros([3, 3, 5, 5]), validate_shape=False)

        g = tf.Graph()
        with tf.Session(graph=g) as session:
            weight_tensor, op = get_model()
            train_op = get_training_step(op)
            prune_ops = prune(weight_tensor)
            saver = tf.train.Saver()
            session.run(tf.global_variables_initializer())

            shape, _, = session.run([tf.shape(op), train_op])
            np.testing.assert_array_equal(shape, [1, 5, 5, 11])

            _ = session.run(prune_ops)
            weights, = session.run([weight_tensor])

            np.testing.assert_array_equal(weights, np.zeros([3, 3, 5, 5]))

            saver.save(session, CHECKPOINT, global_step=1)
            session.close()

        # load the checkpoint file for a model with different size
        starting_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
        assert starting_checkpoint is not None

        #  define the starting network but load the pruned kernels with different shapes/values.
        g = tf.Graph()
        with tf.Session(graph=g) as session:
            weight_tensor, op = get_model()
            train_op = get_training_step(op)
            saver = tf.train.Saver()
            saver.restore(session, starting_checkpoint)

            shape, _, = session.run([tf.shape(op), train_op])

            np.testing.assert_array_equal(shape, [1, 5, 5, 5])
            session.close()


if __name__ == '__main__':
    unittest.main()
