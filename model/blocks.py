import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def relu(input_layer):
    return tf.nn.relu(input_layer)


def conv2d(input_layer, filter_size, input_channels, output_channels, strides=1):
    weights = weight_variable([filter_size, filter_size, input_channels, output_channels])
    logits = tf.nn.conv2d(input_layer, weights, strides=[1, strides, strides, 1], padding="SAME")
    return batch_normalization(logits)


def separable_conv2d(input_layer, filter_size, input_channels, depthwise_multiplier=1, pointwise_multiplier=4):
    intermediate_channels = input_channels * depthwise_multiplier
    depthwise_filter = weight_variable([1, filter_size, input_channels, depthwise_multiplier])

    output_channels = input_channels * pointwise_multiplier
    separable_filter = weight_variable([1, 1, intermediate_channels, output_channels])

    return batch_normalization(
        tf.nn.separable_conv2d(input_layer, depthwise_filter, separable_filter,
                               strides=[1, 1, 1, 1], padding="SAME"))


def residual_bottleneck_separable(input_layer, input_channels, downscaled_outputs, upscaled_outputs, strides=1):
    if input_channels != upscaled_outputs or strides != 1:
        residual = conv2d(input_layer,
                          filter_size=1,
                          input_channels=input_channels,
                          output_channels=upscaled_outputs,
                          strides=strides)
    else:
        residual = input_layer

    activated_input = relu(residual)

    downscaled_features = relu(conv2d(activated_input,
                                      filter_size=1,
                                      input_channels=upscaled_outputs,
                                      output_channels=downscaled_outputs))

    processed_features = relu(separable_conv2d(downscaled_features,
                                               filter_size=3,
                                               input_channels=downscaled_outputs,
                                               depthwise_multiplier=1,
                                               pointwise_multiplier=1))

    upscaled_features = conv2d(processed_features,
                               filter_size=1,
                               input_channels=downscaled_outputs,
                               output_channels=upscaled_outputs)

    return tf.add(upscaled_features, residual)


def batch_normalization(input_layer):
    return tf.nn.batch_normalization(input_layer,
                                     mean=tf.Variable(0, dtype=tf.float32),
                                     variance=tf.Variable(1, dtype=tf.float32),
                                     offset=tf.Variable(0, dtype=tf.float32),
                                     scale=tf.Variable(1, dtype=tf.float32),
                                     variance_epsilon=tf.Variable(1e-8, dtype=tf.float32))


def add_bias(layer, number_of_channels):
    bias = weight_variable([number_of_channels])
    return tf.nn.bias_add(layer, bias)


def fc(input_layer, input_channels, output_channels):
    weights = weight_variable([input_channels, output_channels])
    pre_bias = tf.matmul(input_layer, weights)
    return add_bias(pre_bias, output_channels)
