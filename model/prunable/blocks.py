import tensorflow as tf
from tensorflow.contrib import layers

from model.pruning.op_pruners import ContribLayersBatchNorm, BiasAdd, Matmul, Conv2D, SeparableConv2D, Deconvolution
from model.pruning.relays import ResidualBranch, ResidualConnectionWithStat, EmptyForwardRelay, EmptyBackwardRelay
from model.pruning.stats import ActivationValueStats


class PrunableBlocks:
    def __init__(self, graph_meta, training=True):
        self.graph_meta = graph_meta
        self.training = training

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

    def relu(self, input_layer, pruner):
        relu = tf.nn.relu(input_layer)
        stat = ActivationValueStats(relu)
        stat.set_previous_op(pruner)
        return relu, stat

    def conv2d(self, input_layer, filter_size, input_channels, output_channels, strides, pruner):
        weights = self.weight_variable("conv_weights", [filter_size, filter_size, input_channels, output_channels])
        conv_pruner = Conv2D(weights)
        conv_pruner.set_previous_op(pruner)
        logits = tf.nn.conv2d(input_layer, weights, strides=[1, strides, strides, 1], padding="SAME")
        return self.batch_normalization(logits, conv_pruner)

    def separable_conv2d(self, input_layer, filter_size, input_channels, depthwise_multiplier, pointwise_multiplier,
                         pruner):
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
        separable_pruner = SeparableConv2D(depthwise_filter, pointwise_filter)
        separable_pruner.set_previous_op(pruner)
        return self.batch_normalization(pointwise_results, separable_pruner)

    def residual_bottleneck_separable(self, input_layer, input_channels, downscaled_outputs, upscaled_outputs,
                                      strides, pruner):
        branching_pruner = ResidualBranch()
        branching_pruner.set_previous_op(pruner)
        if input_channels != upscaled_outputs or strides != 1:
            with tf.variable_scope("upscale_residual"):
                residual, residual_pruner = self.conv2d(input_layer,
                                                        filter_size=1,
                                                        input_channels=input_channels,
                                                        output_channels=upscaled_outputs,
                                                        strides=strides,
                                                        pruner=branching_pruner)
                no_relay_pruner = EmptyForwardRelay()
                no_relay_pruner.set_previous_op(residual_pruner)
                residual_pruner = no_relay_pruner

        else:
            residual = input_layer
            residual_pruner = branching_pruner
        pruner = branching_pruner

        with tf.variable_scope("bottleneck_downscale"):
            no_relay_pruner = EmptyBackwardRelay()
            no_relay_pruner.set_previous_op(pruner)
            pruner = no_relay_pruner
            downscaled_features, pruner = self.conv2d(input_layer,
                                                      filter_size=1,
                                                      input_channels=input_channels,
                                                      output_channels=downscaled_outputs,
                                                      strides=strides,
                                                      pruner=pruner)
            downscaled_features, pruner = self.relu(downscaled_features, pruner)
        with tf.variable_scope("bottleneck_convolution"):
            processed_features, pruner = self.separable_conv2d(downscaled_features,
                                                               filter_size=3,
                                                               input_channels=downscaled_outputs,
                                                               depthwise_multiplier=1,
                                                               pointwise_multiplier=1,
                                                               pruner=pruner)
            processed_features, pruner = self.relu(processed_features, pruner)
        with tf.variable_scope("bottleneck_upscale"):
            upscaled_features, pruner = self.conv2d(processed_features,
                                                    filter_size=1,
                                                    input_channels=downscaled_outputs,
                                                    output_channels=upscaled_outputs,
                                                    strides=1,
                                                    pruner=pruner)
            no_relay_pruner = EmptyForwardRelay()
            no_relay_pruner.set_previous_op(pruner)
            pruner = no_relay_pruner

        connection, pruner = self.residual_connection(residual, upscaled_features, residual_pruner, pruner)
        relu, stat_op = self.relu(connection, pruner)
        pruner.set_stat_op(stat_op)
        return relu, pruner

    def residual_connection(self, residual, current, residual_pruner, pruner):
        connection = ResidualConnectionWithStat(residual_pruner, pruner)
        return tf.add(residual, current), connection

    def batch_normalization(self, input_layer, pruner):
        bn = layers.batch_norm(input_layer, fused=True, decay=1.0, scale=True, trainable=self.training)
        bn_pruner = ContribLayersBatchNorm()
        bn_pruner.set_previous_op(pruner)
        return bn, bn_pruner

    def add_bias(self, layer, number_of_channels, pruner):
        bias = self.weight_variable("bias", [number_of_channels])
        bias_add_pruner = BiasAdd(bias)
        bias_add_pruner.set_previous_op(pruner)
        return tf.nn.bias_add(layer, bias), bias_add_pruner

    def fc(self, input_layer, input_channels, output_channels, pruner):
        weights = self.weight_variable("fc_weights", [input_channels, output_channels])
        fc_pruner = Matmul(weights)
        fc_pruner.set_previous_op(pruner)
        return tf.matmul(input_layer, weights), fc_pruner

    def normalized_fc(self, input_layer, input_channels, output_channels, pruner):
        fc, pruner = self.fc(input_layer, input_channels, output_channels, pruner)
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
