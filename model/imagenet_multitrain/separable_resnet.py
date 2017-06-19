import tensorflow as tf

from model.blocks import Blocks
from model.pruning.relays import HeadNode, LastNode


class SeparableResnet:
    def __init__(self, graph_meta=None, training=True, batch_size=None, layer_1_channels=32,
                 training_inputs=None, training_label=None, valid_images=None, valid_label=None):
        """
        :param training: If this is true some update ops and input pipeline will be created.
        The update ops are not necessarily used because training is True.
        """
        self.batch_size = batch_size
        self.blocks = Blocks(graph_meta, training)
        self.training = training
        self.do_validate = tf.placeholder(dtype=tf.bool, shape=None)
        self.layer_1_channels = layer_1_channels
        if training:
            self.training_input, self.training_label = training_inputs, training_label
            self.valid_images, self.valid_label = valid_images, valid_label
            self.input, self.label = tf.cond(self.do_validate,
                                             lambda: (self.valid_images, self.valid_label),
                                             lambda: (self.training_input, self.training_label))

        else:
            self.input = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None, None, 3))
            self.label = tf.placeholder(dtype=tf.int32, shape=(self.batch_size))

        self.last_pruner_node = None
        self.head_pruner_node = None
        self.freeze_layer = None
        self.learning_rate = None

        self.summary_ops = []
        self.logits = self.inference(self.input)

        self.top_1_accuracy = None
        self.top_5_accuracy = None
        self.loss = None
        self.train_step = None
        self.global_step = None
        self.one_hot_truth = None
        self.optimizer = None
        self.optimize()
        self.evaluation()
        self.merged_summary_op = tf.summary.merge(self.summary_ops)

    def inference(self, preprocessed_input):
        channel_multiplier = 1
        channels = self.layer_1_channels * channel_multiplier
        self.head_pruner_node = HeadNode()
        pruner = self.head_pruner_node

        with tf.variable_scope("conv_1_1"):
            conv_1_1, pruner = self.blocks.conv2d(preprocessed_input,
                                                  filter_size=3,
                                                  input_channels=3,
                                                  output_channels=channels,
                                                  strides=2,
                                                  pruner=pruner)
            conv_1_1, pruner = self.blocks.batch_normalization(conv_1_1, pruner)
            conv_1_1, pruner = self.blocks.relu(conv_1_1, pruner)

        with tf.variable_scope("conv_1_2"):
            conv_1_2, pruner = self.blocks.separable_conv2d_with_max_pool(conv_1_1,
                                                                          filter_size=3,
                                                                          input_channels=channels,
                                                                          depthwise_multiplier=2,
                                                                          output_channels=channels * 2,
                                                                          strides=1,
                                                                          pruner=pruner)
        residual_layer = conv_1_2
        channels *= 2
        input_channels = channels * channel_multiplier

        channel_multiplier *= 1
        for residual_block in range(1, 4):
            with tf.variable_scope("conv_2_%d" % residual_block):
                residual_layer, pruner = self.blocks.residual_separable(residual_layer,
                                                                        input_channels=input_channels,
                                                                        output_channels=channels * channel_multiplier,
                                                                        strides=1,
                                                                        activate_before_residual=False
                                                                        if (residual_block == 1) else False,
                                                                        pruner=pruner)
            input_channels = channels * channel_multiplier

        channel_multiplier *= 2
        for residual_block in range(1, 5):
            with tf.variable_scope("conv_3_%d" % residual_block):
                residual_layer, pruner = self.blocks.residual_separable(residual_layer,
                                                                        input_channels=input_channels,
                                                                        output_channels=channels * channel_multiplier,
                                                                        strides=2 if residual_block == 1 else 1,
                                                                        activate_before_residual=False,
                                                                        pruner=pruner)
                input_channels = channels * channel_multiplier

        channel_multiplier *= 2
        for residual_block in range(1, 7):
            with tf.variable_scope("conv_4_%d" % residual_block):
                residual_layer, pruner = self.blocks.residual_separable(residual_layer,
                                                                        input_channels=input_channels,
                                                                        output_channels=channels * channel_multiplier,
                                                                        strides=2 if residual_block == 1 else 1,
                                                                        activate_before_residual=False,
                                                                        pruner=pruner)
            input_channels = channels * channel_multiplier

        channel_multiplier *= 2
        for residual_block in range(1, 4):
            with tf.variable_scope("conv_5_%d" % residual_block):
                residual_layer, pruner = self.blocks.residual_separable(residual_layer,
                                                                        input_channels=input_channels,
                                                                        output_channels=channels * channel_multiplier,
                                                                        strides=2 if residual_block == 1 else 1,
                                                                        activate_before_residual=False,
                                                                        pruner=pruner)
            input_channels = channels * channel_multiplier

        with tf.variable_scope("fc_1"):
            # global average pooling
            residual_layer = tf.reduce_mean(residual_layer, [1, 2])
            # residual_layer, pruner = self.blocks.normalized_fc(residual_layer,
            #                                                    input_channels,
            #                                                    output_channels=input_channels,
            #                                                    pruner=pruner)
            # residual_layer, pruner = self.blocks.relu(residual_layer, pruner)

        with tf.variable_scope("output"):
            logits, pruner = self.blocks.fc(residual_layer,
                                            input_channels=input_channels,
                                            output_channels=1001,
                                            pruner=pruner)
            self.freeze_layer = logits
            self.last_pruner_node = LastNode()
            self.last_pruner_node.set_previous_op(pruner)
        return logits

    def optimize(self):
        with tf.variable_scope('loss'):
            self.one_hot_truth = tf.squeeze(tf.one_hot(self.label, 1001))
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.one_hot_truth)
            self.loss = tf.reduce_mean(cross_entropy)
            self.loss = self.loss + self.decay()
            tf.add_to_collection('losses', self.loss)
            self.summary_ops.append(tf.summary.scalar('loss_total', self.loss))

        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            boundaries = [120000, 250000, 400000, 600000]
            values = [0.01, 0.001, 0.0001, 0.00001, 0.000001]

            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def evaluation(self):
        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), self.label)
            correct_prediction_2 = tf.nn.in_top_k(self.logits, self.label, 5, name=None)
            self.top_1_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.top_5_accuracy = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))
            self.summary_ops.append(tf.summary.scalar('accuracy_top1', self.top_1_accuracy))
            self.summary_ops.append(tf.summary.scalar('accuracy_top5', self.top_5_accuracy))

    def decay(self):
        """L2 weight decay loss."""
        costs = list()
        for var in self.blocks.get_decayed_variables():
            costs.append(tf.nn.l2_loss(var))

        return tf.multiply(0.0001 * (32 / self.layer_1_channels), tf.reduce_sum(costs))
