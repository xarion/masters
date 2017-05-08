# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

import cifar10_input
import tensorflow as tf
from six.moves import urllib

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 6.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial, name=name)


def decomposed_separable_inception_block(input_layer, in_channels, out_channels, filter_sizes=[5],
                                         filter_channel_multipliers=[3], stride=1, padding="SAME"):
    horizontal_results = None
    for (filter_size, channel_multiplier) in zip(filter_sizes, filter_channel_multipliers):
        horizontal_filters = weight_variable([1, filter_size, in_channels, channel_multiplier])
        horizontal_features = depthwise_conv2d(input_layer, horizontal_filters,
                                               strides=[1, stride, stride, 1], padding=padding)
        if horizontal_results is None:
            horizontal_results = horizontal_features
        else:
            horizontal_results = tf.concat([horizontal_results, horizontal_features], axis=3)

    concatenated_channels = sum(filter_channel_multipliers) * in_channels
    horizontal_pointwise_weights = weight_variable([1, 1, concatenated_channels, concatenated_channels])
    mixed_channels = tf.nn.conv2d(horizontal_results, horizontal_pointwise_weights, [1, 1, 1, 1], padding="VALID")
    results = None
    for (filter_size, channel_multiplier) in zip(filter_sizes, filter_channel_multipliers):
        vertical_filters = weight_variable([filter_size, 1, concatenated_channels, channel_multiplier])
        vertical_features = depthwise_conv2d(mixed_channels, vertical_filters,
                                               strides=[1, stride, stride, 1], padding=padding)
        if results is None:
            results = vertical_features
        else:
            results = tf.concat([results, vertical_features], axis=3)

    concatenated_channels *= sum(filter_channel_multipliers)

    pointwise_weights = weight_variable([1, 1, concatenated_channels, out_channels])
    return tf.nn.relu(batch_normalization(tf.nn.conv2d(results, pointwise_weights, [1, 1, 1, 1], padding="VALID")))


def depthwise_conv2d(input_layer, depthwise_filter, strides, padding):
    def op(space_to_batch_input, _, padding):
        return tf.nn.depthwise_conv2d_native(
            input=space_to_batch_input,
            filter=depthwise_filter,
            strides=strides,
            padding=padding,
            name="depthwise")

    return tf.nn.with_space_to_batch(
        input=input_layer,
        filter_shape=tf.shape(depthwise_filter),
        dilation_rate=[1, 1],
        padding=padding,
        op=op)


def decomposed_separable_conv2d(input_layer, filter_size, in_channels, out_channels,
                                channel_multipliers=(2, 2), padding="SAME"):
    intermediate_channels = in_channels * channel_multipliers[0]

    horizontal_depthwise_filter = weight_variable([1, filter_size, in_channels, channel_multipliers[0]])
    horizontal_separable_filter = weight_variable([1, 1, in_channels * channel_multipliers[0], intermediate_channels])
    horizontal = tf.nn.separable_conv2d(input_layer, horizontal_depthwise_filter, horizontal_separable_filter,
                                        strides=[1, 1, 1, 1], padding=padding)
    horizontal = batch_normalization(horizontal)
    horizontal = tf.nn.relu(horizontal)

    vertical_depthwise_filter = weight_variable([filter_size, 1, intermediate_channels, channel_multipliers[1]])
    vertical_separable_filter = weight_variable([1, 1, intermediate_channels * channel_multipliers[1], out_channels])
    vertical = tf.nn.separable_conv2d(horizontal, vertical_depthwise_filter, vertical_separable_filter,
                                      strides=[1, 1, 1, 1], padding=padding)
    vertical = batch_normalization(vertical)
    vertical = tf.nn.relu(vertical)

    return vertical

    # def op(converted_input, _, _padding):
    #     converted_input = tf.nn.depthwise_conv2d_native(converted_input,
    #                                                     filter=weight_variable(
    #                                                         [1, filter_size, in_channels, channel_multipliers[0]],
    #                                                         name="vertical_filter"),
    #                                                     strides=[1, 1, strides, 1],
    #                                                     padding=_padding)
    #     # converted_input = batch_normalization(converted_input)
    #     # converted_input = tf.nn.relu(converted_input)
    #     intermediate_channels = in_channels * channel_multipliers[0]
    #     converted_input = tf.nn.conv2d(converted_input,
    #                                    filter=weight_variable([1, 1, intermediate_channels, intermediate_channels]),
    #                                    strides=[1, 1, 1, 1], padding="VALID")
    #     converted_input = batch_normalization(converted_input)
    #     converted_input = tf.nn.relu(converted_input)
    #
    #     converted_input = tf.nn.depthwise_conv2d_native(converted_input,
    #                                                     filter=weight_variable(
    #                                                         [filter_size, 1, in_channels * channel_multipliers[0],
    #                                                          channel_multipliers[1]], name="horizontal_filter"),
    #                                                     strides=[1, strides, 1, 1],
    #                                                     padding=_padding)
    #     # converted_input = batch_normalization(converted_input)
    #     # converted_input = tf.nn.relu(converted_input)
    #     num_channels = in_channels * channel_multipliers[0] * channel_multipliers[1]
    #     converted_input = tf.nn.conv2d(converted_input,
    #                                    filter=weight_variable([1, 1, num_channels, out_channels]),
    #                                    strides=[1, 1, 1, 1], padding="VALID")
    #     return tf.nn.relu(batch_normalization(converted_input))

    # return tf.nn.with_space_to_batch(input_layer, dilation_rate=[1, 1], padding=padding, op=op)


def batch_normalization(input_layer):
    return tf.nn.batch_normalization(input_layer,
                                     mean=tf.Variable(0, dtype=tf.float32),
                                     variance=tf.Variable(1, dtype=tf.float32),
                                     offset=tf.Variable(0, dtype=tf.float32),
                                     scale=tf.Variable(1, dtype=tf.float32),
                                     variance_epsilon=tf.Variable(1e-8, dtype=tf.float32))


def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)

    return images, labels


def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                                          data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)
    return images, labels


def inference(images):
    conv_1 = decomposed_separable_inception_block(images, in_channels=3, out_channels=32)
    pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                            padding='SAME', name='pool1')
    conv_2 = decomposed_separable_conv2d(pool_1, 5, 64, 64, channel_multipliers=[1, 1], padding="SAME")
    pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # conv_1 = decomposed_separable_conv2d(images, 5, 3, 64, strides=1, channel_multipliers=[4, 4], padding="SAME")
    # pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
    #                         padding='SAME', name='pool1')
    #
    # conv_2 = decomposed_separable_conv2d(pool_1, 5, 64, 64, strides=1, channel_multipliers=[1, 1], padding="SAME")
    #
    # pool2 = tf.nn.max_pool(conv_2, ksize=[1, 3, 3, 1],
    #                        strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    reshape = tf.reshape(pool_2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value

    local3 = tf.nn.relu(batch_normalization(tf.matmul(reshape, weight_variable(shape=[dim, 384]))))
    local4 = tf.nn.relu(batch_normalization(tf.matmul(local3, weight_variable([384, 192]))))

    weights = weight_variable([192, NUM_CLASSES])
    biases = tf.Variable(tf.zeros([NUM_CLASSES]))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases)

    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
  
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
  
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.
  
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
  
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer()
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
