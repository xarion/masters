import unittest

import numpy as np
import tensorflow as tf


class TestTensorflowSeparableWithoutSpaceToBatch(unittest.TestCase):
    def setUp(self):
        # filter height, filter width, in_channels, channel_multiplier
        self.depthwise_weights = np.random.rand(3, 3, 16, 3).astype(np.float32)
        self.pointwise_weights = np.random.rand(1, 1, 48, 64).astype(np.float32)
        # batch size, height, width, channels
        self.input_batch = np.random.rand(20, 224, 224, 16).astype(np.float32)

    def __test_with_configuration(self, strides=1, padding="SAME"):
        with tf.Session() as sess:
            input_placeholder = tf.placeholder(tf.float32, shape=[None, 224, 224, 16])
            _depthwise_weights = tf.Variable(self.depthwise_weights, dtype=tf.float32)

            depthwise_nostb = tf.nn.depthwise_conv2d_native(input=input_placeholder,
                                                            filter=_depthwise_weights,
                                                            strides=[1, strides, strides, 1],
                                                            padding=padding,
                                                            name="depthwise_only")
            # separable without space to batch
            separable_nostb = tf.nn.conv2d(depthwise_nostb,
                                           filter=self.pointwise_weights,
                                           strides=[1, 1, 1, 1],
                                           padding=padding)
            # separable with space to batch
            separable_stb = tf.nn.separable_conv2d(input_placeholder,
                                                   depthwise_filter=self.depthwise_weights,
                                                   pointwise_filter=self.pointwise_weights,
                                                   strides=[1, strides, strides, 1],
                                                   padding=padding)

            sess.run(tf.global_variables_initializer())

            nostb, stb, = sess.run([separable_nostb, separable_stb], feed_dict={input_placeholder: self.input_batch})

            np.testing.assert_array_equal(stb, nostb,
                                          err_msg="output of tensorflow implementation is different than ours")
            sess.close()

    def test_stride_1_padding_same(self):
        self.__test_with_configuration(strides=1, padding="SAME")

    def test_stride_2_padding_same(self):
        self.__test_with_configuration(strides=2, padding="SAME")

    def test_stride_3_padding_same(self):
        self.__test_with_configuration(strides=3, padding="SAME")

    def test_stride_1_padding_valid(self):
        self.__test_with_configuration(strides=1, padding="VALID")

    def test_stride_2_padding_valid(self):
        self.__test_with_configuration(strides=2, padding="VALID")

    def test_stride_3_padding_valid(self):
        self.__test_with_configuration(strides=3, padding="VALID")


if __name__ == '__main__':
    unittest.main()
