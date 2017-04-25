import tensorflow as tf

from blocks import conv2d, relu, residual_bottleneck_separable, fc


class Graph:
    def __init__(self, input_placeholder):
        # self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))
        self.input_placeholder = input_placeholder
        self.logits = self.inference()

    def inference(self):
        with tf.variable_scope("conv_1"):
            conv_1 = conv2d(self.input_placeholder, filter_size=7, input_channels=3, output_channels=64, strides=2)
            relu_1 = relu(conv_1)

        with tf.variable_scope("max_pool_1"):
            max_pool_1 = tf.nn.max_pool(relu_1, [1, 3, 3, 1], [1, 2, 2, 1], padding="SAME")

        residual_layer = max_pool_1

        input_channels = 64
        for residual_block in range(1, 4):
            with tf.variable_scope("conv_2_%d" % residual_block):
                residual_layer = residual_bottleneck_separable(residual_layer,
                                                               input_channels=input_channels,
                                                               downscaled_outputs=64,
                                                               upscaled_outputs=256,
                                                               strides=1)
            input_channels = 256

        for residual_block in range(1, 5):
            with tf.variable_scope("conv_3_%d" % residual_block):
                residual_layer = residual_bottleneck_separable(residual_layer,
                                                               input_channels=input_channels,
                                                               downscaled_outputs=128,
                                                               upscaled_outputs=512,
                                                               strides=2 if residual_block == 1 else 1)
            input_channels = 512

        for residual_block in range(1, 7):
            with tf.variable_scope("conv_4_%d" % residual_block):
                residual_layer = residual_bottleneck_separable(residual_layer,
                                                               input_channels=input_channels,
                                                               downscaled_outputs=256,
                                                               upscaled_outputs=1024,
                                                               strides=2 if residual_block == 1 else 1)
            input_channels = 1024

        for residual_block in range(1, 4):
            with tf.variable_scope("conv_5_%d" % residual_block):
                residual_layer = residual_bottleneck_separable(residual_layer,
                                                               input_channels=input_channels,
                                                               downscaled_outputs=512,
                                                               upscaled_outputs=2048,
                                                               strides=2 if residual_block == 1 else 1)
            input_channels = 2048

        with tf.variable_scope("fully_connected"):
            avg_pool = tf.nn.avg_pool(residual_layer, [1, 7, 7, 1], [1, 1, 1, 1], "VALID")
            avg_pool = tf.squeeze(avg_pool)
            logits = fc(avg_pool, input_channels=2048, output_channels=1000)

        return logits
