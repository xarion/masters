import tensorflow as tf

import data as cifar10
from model.blocks import Blocks
from model.pruning.relays import HeadNode, LastNode


class SeparableResnet:
    def __init__(self, graph_meta=None, training=True, batch_size=None):
        """
        :param training: If this is true some update ops and input pipeline will be created.
        The update ops are not necessarily used because training is True.
        """
        self.batch_size = batch_size
        self.blocks = Blocks(graph_meta, training)
        self.training = training
        self.do_validate = tf.placeholder(dtype=tf.bool, shape=None)
        if training:
            with tf.device("/cpu:0"):
                self.training_input, self.training_label = cifar10.distorted_inputs("data/cifar-10-batches-bin",
                                                                                    self.batch_size)
                self.valid_images, self.valid_label = cifar10.inputs(True, "data/cifar-10-batches-bin",
                                                                     self.batch_size)
            self.input, self.label = tf.cond(self.do_validate,
                                             lambda: (self.valid_images, self.valid_label),
                                             lambda: (self.training_input, self.training_label))
        else:
            self.input = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 24, 24, 3))
            self.label = tf.placeholder(dtype=tf.int32, shape=(self.batch_size))

        # tf.summary.image("input_image", self.input)
        # with tf.device("/cpu:0"):
        #     preprocessed_input = self.pre_process_input()
        self.last_pruner_node = None
        self.head_pruner_node = None
        self.freeze_layer = None
        self.learning_rate = None
        self.logits = self.inference(self.input)

        self.top_1_accuracy = None
        self.loss = None
        self.train_step = None
        self.global_step = None
        self.one_hot_truth = None
        self.optimizer = None
        self.optimize()
        self.evaluation()
        # dummy variable that is temporarily ignored

    # def pre_process_input(self):
    #     norm = tf.div(self.input, tf.constant(255.0 / 2), 'norm')
    #     return tf.subtract(norm, tf.constant(1.), 'trans')

    def inference(self, preprocessed_input):
        self.head_pruner_node = HeadNode()
        pruner = self.head_pruner_node
        with tf.variable_scope("conv_1"):
            conv_1, pruner = self.blocks.conv2d(preprocessed_input,
                                                filter_size=3,
                                                input_channels=3,
                                                output_channels=16,
                                                strides=1,
                                                pruner=pruner)
        residual_layer = conv_1

        input_channels = 16
        for residual_block in range(1, 4):
            with tf.variable_scope("conv_2_%d" % residual_block):
                residual_layer, pruner = self.blocks.residual_separable(residual_layer,
                                                                        input_channels=input_channels,
                                                                        output_channels=32,
                                                                        strides=1,
                                                                        activate_before_residual=True
                                                                        if (residual_block == 1) else False,
                                                                        pruner=pruner)
            input_channels = 32

        for residual_block in range(1, 5):
            with tf.variable_scope("conv_3_%d" % residual_block):
                residual_layer, pruner = self.blocks.residual_separable(residual_layer,
                                                                        input_channels=input_channels,
                                                                        output_channels=64,
                                                                        strides=1,
                                                                        activate_before_residual=False,
                                                                        pruner=pruner)
            input_channels = 64

        for residual_block in range(1, 7):
            with tf.variable_scope("conv_4_%d" % residual_block):
                residual_layer, pruner = self.blocks.residual_separable(residual_layer,
                                                                        input_channels=input_channels,
                                                                        output_channels=128,
                                                                        strides=2 if residual_block == 1 else 1,
                                                                        activate_before_residual=False,
                                                                        pruner=pruner)
            input_channels = 128

        for residual_block in range(1, 4):
            with tf.variable_scope("conv_5_%d" % residual_block):
                residual_layer, pruner = self.blocks.residual_separable(residual_layer,
                                                                        input_channels=input_channels,
                                                                        output_channels=256,
                                                                        strides=2 if residual_block == 1 else 1,
                                                                        activate_before_residual=False,
                                                                        pruner=pruner)
            input_channels = 256

        with tf.variable_scope("output"):
            residual_layer, pruner = self.blocks.batch_normalization(residual_layer, pruner)
            residual_layer, pruner = self.blocks.relu(residual_layer, pruner)
            # global average pooling
            residual_layer = tf.reduce_mean(residual_layer, [1, 2])

            logits, pruner = self.blocks.biased_fc(residual_layer,
                                                   input_channels=input_channels,
                                                   output_channels=10,
                                                   pruner=pruner)
            self.last_pruner_node = LastNode()
            self.last_pruner_node.set_previous_op(pruner)
        return logits

    def optimize(self):
        with tf.variable_scope('loss'):
            self.one_hot_truth = tf.squeeze(tf.one_hot(self.label, 10))
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.one_hot_truth)
            self.loss = tf.reduce_mean(cross_entropy)
            self.loss = self.loss + self.decay()
            tf.add_to_collection('losses', self.loss)
            tf.summary.scalar('loss_total', self.loss)

        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            boundaries = [40000, 60000, 80000]
            values = [0.1, 0.01, 0.001, 0.0001]

            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
            # self.optimizer = tf.train.AdamOptimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def evaluation(self):
        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), self.label)
            correct_prediction_2 = tf.nn.in_top_k(self.logits, self.label, 5, name=None)
            self.top_1_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            top_5_accuracy = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))
            tf.summary.scalar('accuracy_top1', self.top_1_accuracy)
            tf.summary.scalar('accuracy_top5', top_5_accuracy)

    def decay(self):
        """L2 weight decay loss."""
        costs = list()
        for var in tf.trainable_variables():
            if var.op.name.find(r'_weights') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(0.0002, tf.reduce_sum(costs))
