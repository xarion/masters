import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class Blocks:
    def __init__(self, graph_meta):
        self.graph_meta = graph_meta

    def get_variable_with_shape_by_graph_meta_getter(self, getter, name, shape, *args, **kwargs):
        if self.graph_meta is not None:
            shape_from_meta = self.graph_meta.get_variable_shape(name)
            if shape_from_meta is not None:
                shape = shape_from_meta
            else:
                self.graph_meta.set_variable_shape(name, shape)
        return getter(name, shape, *args, **kwargs)

    def weight_variable(self, name, shape):
        return tf.get_variable(name,
                               shape=shape,
                               initializer=xavier_initializer(),
                               custom_getter=self.get_variable_with_shape_by_graph_meta_getter)

    def relu(self, input_layer):
        return tf.nn.relu(input_layer)

    def conv2d(self, input_layer, filter_size, input_channels, output_channels, strides=1):
        weights = self.weight_variable("conv_weights", [filter_size, filter_size, input_channels, output_channels])
        logits = tf.nn.conv2d(input_layer, weights, strides=[1, strides, strides, 1], padding="SAME")
        return self.batch_normalization(logits)

    def separable_conv2d(self, input_layer, filter_size, input_channels, depthwise_multiplier=1,
                         pointwise_multiplier=1):
        intermediate_channels = input_channels * depthwise_multiplier
        depthwise_filter = self.weight_variable("depthwise_filter",
                                                [filter_size, filter_size, input_channels, depthwise_multiplier])

        output_channels = input_channels * pointwise_multiplier
        pointwise_filter = self.weight_variable("pointwise_filter", [1, 1, intermediate_channels, output_channels])

        depthwise_results = tf.nn.depthwise_conv2d_native(input=input_layer,
                                                          filter=depthwise_filter,
                                                          strides=[1, 1, 1, 1],
                                                          padding="SAME",
                                                          name="depthwise_2")
        pointwise_results = tf.nn.conv2d(depthwise_results,
                                         pointwise_filter,
                                         [1, 1, 1, 1],
                                         padding="VALID")

        return self.batch_normalization(pointwise_results)

    def residual_bottleneck_separable(self, input_layer, input_channels, downscaled_outputs, upscaled_outputs,
                                      strides=1):
        if input_channels != upscaled_outputs or strides != 1:
            with tf.variable_scope("upscale_residual"):
                residual = self.conv2d(input_layer,
                                       filter_size=1,
                                       input_channels=input_channels,
                                       output_channels=upscaled_outputs,
                                       strides=strides)
        else:
            residual = input_layer

        activated_input = self.relu(residual)
        with tf.variable_scope("bottleneck_downscale"):
            downscaled_features = self.relu(self.conv2d(activated_input,
                                                        filter_size=1,
                                                        input_channels=upscaled_outputs,
                                                        output_channels=downscaled_outputs))
        with tf.variable_scope("bottleneck_convolution"):
            processed_features = self.relu(self.separable_conv2d(downscaled_features,
                                                                 filter_size=3,
                                                                 input_channels=downscaled_outputs,
                                                                 depthwise_multiplier=1,
                                                                 pointwise_multiplier=1))
        with tf.variable_scope("bottleneck_upscale"):
            upscaled_features = self.conv2d(processed_features,
                                            filter_size=1,
                                            input_channels=downscaled_outputs,
                                            output_channels=upscaled_outputs)

        return tf.add(upscaled_features, residual)

    def batch_normalization(self, input_layer):
        return tf.nn.batch_normalization(input_layer,
                                         mean=tf.Variable(0, dtype=tf.float32),
                                         variance=tf.Variable(1, dtype=tf.float32),
                                         offset=tf.Variable(0, dtype=tf.float32),
                                         scale=tf.Variable(1, dtype=tf.float32),
                                         variance_epsilon=tf.Variable(1e-8, dtype=tf.float32))

    def add_bias(self, layer, number_of_channels):
        bias = self.weight_variable("bias", [number_of_channels])
        return tf.nn.bias_add(layer, bias)

    def normalized_fc(self, input_layer, input_channels, output_channels):
        weights = self.weight_variable("fc_weights", [input_channels, output_channels])
        return self.batch_normalization(tf.matmul(input_layer, weights))

    def biased_fc(self, input_layer, input_channels, output_channels):
        weights = self.weight_variable("fc_weights", [input_channels, output_channels])
        pre_bias = tf.matmul(input_layer, weights)
        return self.add_bias(pre_bias, output_channels)
