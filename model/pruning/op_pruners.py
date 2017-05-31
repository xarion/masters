import tensorflow as tf

from . import OpPruning


class Conv2D(OpPruning):
    def __init__(self, weight_tensor):
        OpPruning.__init__(self)
        self.weight_tensor = weight_tensor

    def prune_input_channel(self, keep_indices):
        self.weight_tensor = self.prune(self.weight_tensor, keep_indices, axis=2)

    def prune_output_channel(self, keep_indices):
        self.weight_tensor = self.prune(self.weight_tensor, keep_indices, axis=3)


class SeparableConv2D(OpPruning):
    def __init__(self, depthwise_weight_tensor, pointwise_weight_tensor):
        OpPruning.__init__(self)
        self.depthwise_weight_tensor = depthwise_weight_tensor
        self.pointwise_weight_tensor = pointwise_weight_tensor

    def prune_input_channel(self, keep_indices):
        self.depthwise_weight_tensor = self.prune(self.depthwise_weight_tensor, keep_indices, axis=2)
        self.pointwise_weight_tensor = self.prune(self.pointwise_weight_tensor, keep_indices, axis=2)

    def prune_output_channel(self, keep_indices):
        self.pointwise_weight_tensor = self.prune(self.pointwise_weight_tensor, keep_indices, axis=3)


class ContribLayersBatchNorm(OpPruning):
    def __init__(self):
        """
        Assumes that we are creating an instance within the same name scope as the batch_norm, after batch_norm.
        """
        OpPruning.__init__(self)

        with tf.variable_scope("BatchNorm", reuse=True):
            self.beta = tf.get_variable('beta')
            self.moving_mean = tf.get_variable('moving_mean')
            self.moving_variance = tf.get_variable('moving_variance')
            try:
                self.gamma = tf.get_variable('gamma')
            except Exception:
                #  gamma may not exist if fused=False
                self.gamma = None

    def prune_input_channel(self, keep_indices):
        pass

    def prune_output_channel(self, keep_indices):
        self.beta = self.prune(self.beta, keep_indices, axis=0)
        self.moving_mean = self.prune(self.moving_mean, keep_indices, axis=0)
        self.moving_variance = self.prune(self.moving_variance, keep_indices, axis=0)
        if self.gamma:
            self.gamma = self.prune(self.gamma, keep_indices, axis=0)


class BiasAdd(OpPruning):
    def __init__(self, bias):
        OpPruning.__init__(self)
        self.bias = bias

    def prune_input_channel(self, keep_indices):
        return []

    def prune_output_channel(self, keep_indices):
        self.bias = self.prune(self.bias, keep_indices, axis=0)


class Matmul(OpPruning):
    def __init__(self, weight_tensor):
        OpPruning.__init__(self)
        self.weight_tensor = weight_tensor

    def prune_input_channel(self, keep_indices):
        self.weight_tensor = self.prune(self.weight_tensor, keep_indices, axis=0)

    def prune_output_channel(self, keep_indices):
        self.weight_tensor = self.prune(self.weight_tensor, keep_indices, axis=1)


class ResidualLayerOutput(OpPruning):
    """
     Do nothing right now, assign does not work in this case. we have to insert an op between two ops.
     And this adds the requirement of modifying the graph in another level.
     should use tf.scatter_nd_add() probably with combination of other tf.tile and reshape.
    """

    def prune_output_channel(self, keep_indices):
        pass

    def prune_input_channel(self, keep_indices):
        pass


class Deconvolution(OpPruning):
    def __init__(self, weight_tensor):
        OpPruning.__init__(self)
        self.weight_tensor = weight_tensor

    def prune_input_channel(self, keep_indices):
        self.weight_tensor = self.prune(self.weight_tensor, keep_indices, axis=3)

    def prune_output_channel(self, keep_indices):
        self.weight_tensor = self.prune(self.weight_tensor, keep_indices, axis=2)
