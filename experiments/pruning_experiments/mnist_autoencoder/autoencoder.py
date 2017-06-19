import tensorflow as tf

from model.blocks import Blocks
from model.pruning.relays import HeadNode, LastNode
from model.pruning.stats import ActivationValueStats


class Autoencoder:
    def __init__(self, graph_meta=None, training=True, batch_size=None):
        """
        :param training: If this is true some update ops and input pipeline will be created.
        The update ops are not necessarily used because training is True.
        """
        self.batch_size = batch_size
        self.blocks = Blocks(graph_meta, training)
        self.training = training
        self.do_validate = tf.placeholder(dtype=tf.bool, shape=None)

        self.input = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 28*28))

        self.reshaped_input = tf.reshape(self.input, [self.batch_size, 28, 28, 1])
        self.last_pruner_node = None
        self.head_pruner_node = None
        self.autoencode()

        self.loss = None
        self.train_step = None
        self.global_step = None

        self.optimizer = None
        self.optimize()
        # dummy variable that is temporarily ignored

    # def pre_process_input(self):
    #     norm = tf.div(self.input, tf.constant(255.0 / 2), 'norm')
    #     return tf.subtract(norm, tf.constant(1.), 'trans')

    def autoencode(self):
        self.head_pruner_node = HeadNode()
        pruner = self.head_pruner_node

        with tf.variable_scope("encoder_1"):
            encoder_1, pruner = self.blocks.conv2d(self.reshaped_input,
                                                   filter_size=3,
                                                   input_channels=1,
                                                   output_channels=32,
                                                   strides=2,
                                                   pruner=pruner)

            encoder_1, pruner = self.blocks.batch_normalization(encoder_1, pruner)
            encoder_1, pruner = self.blocks.relu(encoder_1, pruner)

        with tf.variable_scope("encoder_2"):
            encoder_2, pruner = self.blocks.conv2d(encoder_1,
                                                   filter_size=3,
                                                   input_channels=32,
                                                   output_channels=64,
                                                   strides=2,
                                                   pruner=pruner)
            encoder_2, pruner = self.blocks.batch_normalization(encoder_2, pruner)
            encoder_2, pruner = self.blocks.relu(encoder_2, pruner)

        with tf.variable_scope("decoder_1"):
            decoder_1, pruner = self.blocks.deconvolution(encoder_2,
                                                          filter_size=3,
                                                          input_channels=64,
                                                          output_channels=32,
                                                          output_dimensions=[self.batch_size, 14, 14, 32],
                                                          strides=2,
                                                          pruner=pruner)
            decoder_1, pruner = self.blocks.batch_normalization(decoder_1, pruner)
            decoder_1, pruner = self.blocks.relu(decoder_1, pruner)

        with tf.variable_scope("decoder_2"):
            decoder_2, pruner = self.blocks.deconvolution(decoder_1,
                                                          filter_size=3,
                                                          input_channels=32,
                                                          output_channels=1,
                                                          output_dimensions=[self.batch_size, 28, 28, 1],
                                                          strides=2,
                                                          pruner=pruner)

        with tf.variable_scope("output"):
            self.output = tf.nn.tanh(decoder_2)
            self.last_pruner_node = LastNode()
            self.last_pruner_node.set_previous_op(pruner)
        return self.output

    def optimize(self):
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.output, self.reshaped_input))

            tf.add_to_collection('losses', self.loss)
            tf.summary.scalar('loss_total', self.loss)

        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            boundaries = [400, 600, 700]
            values = [0.1, 0.01, 0.001, 0.0001]

            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
            #
            # self.optimizer = tf.train.AdamOptimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def relu(self, input_layer, pruner):
        relu = tf.nn.relu(input_layer)
        stat = ActivationValueStats(relu)
        stat.set_previous_op(pruner)
        return relu, stat