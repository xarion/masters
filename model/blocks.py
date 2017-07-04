import tensorflow as tf
from tensorflow.contrib import layers

from model.pruning.op_pruners import ContribLayersBatchNorm, BiasAdd, Matmul, Conv2D, SeparableConv2D, Deconvolution, \
    FeaturePadding
from model.pruning.relays import ResidualBranch, EmptyForwardRelay, EmptyBackwardRelay, \
    ResidualJoin
from model.pruning.stats import ActivationCorrelationStats


class Blocks:
    def __init__(self, graph_meta, training=True):
        self.graph_meta = graph_meta
        self.training = training
        self.decayed_variables = []

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
                               initializer=layers.xavier_initializer(),
                               custom_getter=self.get_variable_with_shape_by_graph_meta_getter)

    def weight_variable_with_initializer(self, name, shape, initializer):
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               custom_getter=self.get_variable_with_shape_by_graph_meta_getter)

    def relu(self, input_layer, pruner):
        relu = tf.nn.relu(input_layer)
        stat = ActivationCorrelationStats(relu)
        stat.set_previous_op(pruner)
        return relu, stat

    def conv2d(self, input_layer, filter_size, input_channels, output_channels, strides, pruner):
        weights = self.weight_variable("conv_weights", [filter_size, filter_size, input_channels, output_channels])
        self.decayed_variables.append(weights)
        conv_pruner = Conv2D(weights)
        conv_pruner.set_previous_op(pruner)
        logits = tf.nn.conv2d(input_layer, weights, strides=[1, strides, strides, 1], padding="SAME")
        logits, pruner = self.add_bias(logits, output_channels, pruner)
        return logits, conv_pruner

    def composed_conv2d(self, input_layer, filter_size, input_channels, output_channels, intermediate_channels,
                        strides, pruner):
        weights_1 = self.weight_variable("1d_weights_1", [1, filter_size, input_channels, intermediate_channels])
        self.decayed_variables.append(weights_1)
        conv_pruner = Conv2D(weights_1)
        conv_pruner.set_previous_op(pruner)
        logits = tf.nn.conv2d(input_layer, weights_1, strides=[1, 1, strides, 1], padding="SAME")
        logits, pruner = self.batch_normalization(logits, pruner)
        logits, pruner = self.relu(logits, pruner)
        weights_2 = self.weight_variable("1d_weights_2", [filter_size, 1, intermediate_channels, output_channels])
        self.decayed_variables.append(weights_2)
        conv_pruner = Conv2D(weights_2)
        conv_pruner.set_previous_op(pruner)
        logits = tf.nn.conv2d(logits, weights_2, strides=[1, strides, 1, 1], padding="SAME")
        return logits, conv_pruner

    def separable_conv2d(self, input_layer, filter_size, input_channels, depthwise_multiplier, output_channels,
                         strides, pruner, depthwise_relu_bn=False):
        intermediate_channels = input_channels * depthwise_multiplier

        with tf.variable_scope('depthwise'):
            depthwise_weights = self.weight_variable("depthwise_weights",
                                                     [filter_size, filter_size, input_channels, depthwise_multiplier])
            self.decayed_variables.append(depthwise_weights)
            depthwise_results = tf.nn.depthwise_conv2d_native(input=input_layer,
                                                              filter=depthwise_weights,
                                                              strides=[1, strides, strides, 1],
                                                              padding="SAME",
                                                              name="depthwise")
            # depthwise_results, pruner = self.add_bias(depthwise_results, intermediate_channels, pruner)

            if depthwise_relu_bn:
                depthwise_results, pruner = self.batch_normalization(depthwise_results, pruner)
                depthwise_results, pruner = self.relu(depthwise_results, pruner)

        with tf.variable_scope('pointwise'):
            pointwise_weights = self.weight_variable("pointwise_weights",
                                                     [1, 1, intermediate_channels, output_channels])
            self.decayed_variables.append(pointwise_weights)
            pointwise_results = tf.nn.conv2d(depthwise_results,
                                             pointwise_weights,
                                             [1, 1, 1, 1],
                                             padding="VALID")
            pointwise_results, pruner = self.add_bias(pointwise_results, output_channels, pruner)

        separable_pruner = SeparableConv2D(depthwise_weights, pointwise_weights)
        separable_pruner.set_previous_op(pruner)
        return pointwise_results, separable_pruner

    def residual_separable(self, input_layer, input_channels, output_channels,
                           strides, activate_before_residual, pruner):

        if activate_before_residual:
            input_layer, pruner = self.batch_normalization(input_layer, pruner)
            input_layer, pruner = self.relu(input_layer, pruner)
            residual = input_layer
            branching_pruner = ResidualBranch()
            branching_pruner.set_previous_op(pruner)
            pruner = branching_pruner
        else:
            branching_pruner = ResidualBranch()
            branching_pruner.set_previous_op(pruner)
            pruner = branching_pruner
            residual = input_layer
            input_layer, pruner = self.batch_normalization(input_layer, pruner)
            input_layer, pruner = self.relu(input_layer, pruner)

        empty_backward_relay = EmptyBackwardRelay()
        empty_backward_relay.set_previous_op(pruner)
        pruner = empty_backward_relay
        with tf.variable_scope("convolution_1"):
            features, pruner = self.separable_conv2d(input_layer,
                                                     filter_size=3,
                                                     input_channels=input_channels,
                                                     depthwise_multiplier=1,
                                                     output_channels=output_channels,
                                                     strides=strides,
                                                     pruner=pruner)

        with tf.variable_scope("convolution_2"):
            features, pruner = self.batch_normalization(features, pruner)
            features, pruner = self.relu(features, pruner)
            features, pruner = self.separable_conv2d(features,
                                                     filter_size=3,
                                                     input_channels=output_channels,
                                                     depthwise_multiplier=1,
                                                     output_channels=output_channels,
                                                     strides=1,
                                                     pruner=pruner)
            empty_forward_relay = EmptyForwardRelay()
            empty_forward_relay.set_previous_op(pruner)
            pruner = empty_forward_relay

        with tf.variable_scope("residual_connection"):
            if strides is not 1 or input_channels != output_channels:
                empty_backward_relay = EmptyBackwardRelay()
                empty_backward_relay.set_previous_op(branching_pruner)
                residual = tf.nn.avg_pool(residual,
                                          ksize=[1, strides, strides, 1],
                                          strides=[1, strides, strides, 1],
                                          padding="VALID")

                residual, branching_pruner = self.pad_residual_features(residual,
                                                                        input_channels,
                                                                        output_channels,
                                                                        empty_backward_relay)
                empty_forward_relay = EmptyForwardRelay()
                empty_forward_relay.set_previous_op(branching_pruner)
                branching_pruner = empty_forward_relay
            connection, pruner = self.residual_connection(residual, features, branching_pruner, pruner)

        return connection, pruner

    def pad_residual_features(self, residual, input_channels, output_channels, pruner):
        half_channel_difference = (output_channels - input_channels) // 2

        beginning_pad_count = tf.Variable(initial_value=half_channel_difference,
                                          name="beginning_pad_count")
        ending_pad_count = tf.Variable(initial_value=output_channels - (input_channels + half_channel_difference),
                                       name="ending_pad_count")
        residual = tf.pad(residual,
                          [[0, 0], [0, 0], [0, 0], [beginning_pad_count, ending_pad_count]])
        padding_pruner = FeaturePadding(beginning_pad_count, ending_pad_count, input_channels)
        padding_pruner.set_previous_op(pruner)
        return residual, padding_pruner

    def residual_connection(self, residual, current, residual_pruner, pruner):
        connection = ResidualJoin()
        connection.set_previous_op(residual_pruner)
        connection.set_previous_op(pruner)
        return tf.add(residual, current), connection

    def batch_normalization(self, input_layer, pruner):
        bn = layers.batch_norm(input_layer, fused=True, trainable=self.training, scale=True)
        bn_pruner = ContribLayersBatchNorm()
        bn_pruner.set_previous_op(pruner)
        return bn, bn_pruner

    def add_bias(self, layer, number_of_channels, pruner):
        bias = self.weight_variable_with_initializer("bias", [number_of_channels], tf.zeros_initializer())
        bias_add_pruner = BiasAdd(bias)
        bias_add_pruner.set_previous_op(pruner)
        return tf.nn.bias_add(layer, bias), bias_add_pruner

    def fc(self, input_layer, input_channels, output_channels, pruner):
        weights = self.weight_variable("fc_weights", [input_channels, output_channels])
        self.decayed_variables.append(weights)
        fc_pruner = Matmul(weights)

        fc_pruner.set_previous_op(pruner)
        return tf.matmul(input_layer, weights), fc_pruner

    def normalized_fc(self, input_layer, input_channels, output_channels, pruner):
        fc, pruner = self.biased_fc(input_layer, input_channels, output_channels, pruner)
        fc = tf.expand_dims(fc, 1)
        fc = tf.expand_dims(fc, 1)
        bn, pruner = self.batch_normalization(fc, pruner)
        return tf.squeeze(bn, axis=[1, 2]), pruner

    def biased_fc(self, input_layer, input_channels, output_channels, pruner):
        pre_bias, pruner = self.fc(input_layer, input_channels, output_channels, pruner)
        return self.add_bias(pre_bias, output_channels, pruner)

    def deconvolution(self, input_layer, filter_size, input_channels, output_channels, output_dimensions, strides,
                      pruner):
        weights = self.weight_variable("deconv_weights", [filter_size, filter_size, output_channels, input_channels])
        self.decayed_variables.append(weights)
        output_shape = [output_dimensions[0],
                        output_dimensions[1],
                        output_dimensions[2],
                        weights.get_shape()[2].value]
        deconv = tf.nn.conv2d_transpose(input_layer,
                                        weights,
                                        output_shape,
                                        strides=[1, strides, strides, 1],
                                        padding="SAME")
        deconv_pruner = Deconvolution(weights)
        deconv_pruner.set_previous_op(pruner)

        return deconv, deconv_pruner

    def separable_conv2d_with_max_pool(self, input_layer, filter_size, input_channels, depthwise_multiplier,
                                       output_channels, strides, pruner):
        intermediate_channels = input_channels * depthwise_multiplier
        with tf.variable_scope('depthwise'):
            depthwise_weights = self.weight_variable("depthwise_weights",
                                                     [filter_size, filter_size, input_channels, depthwise_multiplier])

            pointwise_weights = self.weight_variable("pointwise_weights", [1, 1, intermediate_channels, output_channels])
            self.decayed_variables.append(depthwise_weights)
            self.decayed_variables.append(pointwise_weights)
            depthwise_results = tf.nn.depthwise_conv2d_native(input=input_layer,
                                                              filter=depthwise_weights,
                                                              strides=[1, strides, strides, 1],
                                                              padding="SAME",
                                                              name="depthwise")
            # depthwise_results, pruner = self.add_bias(depthwise_results, intermediate_channels, pruner)
            depthwise_results_max = tf.nn.max_pool(depthwise_results, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding="SAME")
            depthwise_results_avg = tf.nn.avg_pool(depthwise_results, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                   padding="SAME")
            depthwise_results = depthwise_results_avg + depthwise_results_max
        with tf.variable_scope('pointwise'):
            pointwise_results = tf.nn.conv2d(depthwise_results,
                                             pointwise_weights,
                                             [1, 1, 1, 1],
                                             padding="VALID")
            pointwise_results, pruner = self.add_bias(pointwise_results, output_channels, pruner)
        separable_pruner = SeparableConv2D(depthwise_weights, pointwise_weights)
        separable_pruner.set_previous_op(pruner)
        return pointwise_results, separable_pruner

    def get_decayed_variables(self):
        return self.decayed_variables
