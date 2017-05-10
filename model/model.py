import tensorflow as tf

from blocks import Blocks
from graph_meta import GraphMeta


class SeparableResnet:
    def __init__(self, learning_rate=None, input_tensor=None, label_tensor=None, graph_meta=None):
        self.blocks = Blocks(graph_meta)
        if input_tensor is None:
            self.input = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))
        else:
            self.input = input_tensor

        if label_tensor is None:
            self.label = tf.placeholder(dtype=tf.int32, shape=(None))
        else:
            self.label = label_tensor

        preprocessed_input = self.pre_process_input()

        self.freeze_layer = None
        self.learning_rate = learning_rate
        self.logits = self.inference(preprocessed_input)

        self.top_1_accuracy = None
        self.loss = None
        self.train_step = None
        self.global_step = None
        self.one_hot_truth = None
        self.training()
        self.evaluation()
        # dummy variable that is temporarily ignored
        self.inf = tf.placeholder_with_default(True, None, name='inference')
        self.keep_prob = tf.placeholder(tf.float32)
        self.pop_vars = tf.Variable(0)

    def pre_process_input(self):
        norm = tf.div(self.input, tf.constant(255.0), 'norm')
        return tf.subtract(norm, tf.constant(0.5), 'trans')

    def inference(self, preprocessed_input):

        with tf.variable_scope("conv_1"):
            conv_1 = self.blocks.conv2d(preprocessed_input,
                                        filter_size=7,
                                        input_channels=3,
                                        output_channels=64,
                                        strides=2)
            relu_1 = self.blocks.relu(conv_1)

        with tf.variable_scope("max_pool_1"):
            max_pool_1 = tf.nn.max_pool(relu_1, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")

        residual_layer = max_pool_1

        input_channels = 64
        for residual_block in range(1, 4):
            with tf.variable_scope("conv_2_%d" % residual_block):
                residual_layer = self.blocks.residual_bottleneck_separable(residual_layer,
                                                                           input_channels=input_channels,
                                                                           downscaled_outputs=64,
                                                                           upscaled_outputs=256,
                                                                           strides=1)
            input_channels = 256

        for residual_block in range(1, 5):
            with tf.variable_scope("conv_3_%d" % residual_block):
                residual_layer = self.blocks.residual_bottleneck_separable(residual_layer,
                                                                           input_channels=input_channels,
                                                                           downscaled_outputs=128,
                                                                           upscaled_outputs=512,
                                                                           strides=2 if residual_block == 1 else 1)
            input_channels = 512

        for residual_block in range(1, 7):
            with tf.variable_scope("conv_4_%d" % residual_block):
                residual_layer = self.blocks.residual_bottleneck_separable(residual_layer,
                                                                           input_channels=input_channels,
                                                                           downscaled_outputs=256,
                                                                           upscaled_outputs=1024,
                                                                           strides=2 if residual_block == 1 else 1)
            input_channels = 1024

        for residual_block in range(1, 4):
            with tf.variable_scope("conv_5_%d" % residual_block):
                residual_layer = self.blocks.residual_bottleneck_separable(residual_layer,
                                                                           input_channels=input_channels,
                                                                           downscaled_outputs=512,
                                                                           upscaled_outputs=2048,
                                                                           strides=2 if residual_block == 1 else 1)
            input_channels = 2048

        with tf.variable_scope("fully_connected_1"):
            avg_pool = tf.nn.avg_pool(residual_layer, [1, 7, 7, 1], [1, 1, 1, 1], "VALID")

            avg_pool = tf.squeeze(avg_pool, axis=[1, 2])
            self.freeze_layer = avg_pool

            fc1 = self.blocks.normalized_fc(avg_pool,
                                            input_channels=2048,
                                            output_channels=1000)
        with tf.variable_scope("output"):
            logits = self.blocks.biased_fc(fc1,
                                           input_channels=1000,
                                           output_channels=1000)
        return logits

    def training(self):
        with tf.variable_scope('loss'):
            self.one_hot_truth = tf.squeeze(tf.one_hot(self.label, 1000, on_value=0.9, off_value=0.0001))
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.one_hot_truth)
            self.loss = tf.reduce_mean(cross_entropy)
            tf.add_to_collection('losses', self.loss)

        with tf.variable_scope('train'):
            tf.summary.scalar('loss_total', self.loss)
            if self.learning_rate is not None:
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            else:
                optimizer = tf.train.AdamOptimizer()
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)

    def evaluation(self):
        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), self.label)
            correct_prediction_2 = tf.nn.in_top_k(self.logits, self.label, 5, name=None)
            self.top_1_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            top_5_accuracy = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))
            tf.summary.scalar('accuracy_top1', self.top_1_accuracy)
            tf.summary.scalar('accuracy_top5', top_5_accuracy)
