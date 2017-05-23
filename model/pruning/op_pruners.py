import tensorflow as tf

from . import OpPruning


class Conv2D(OpPruning):
    def __init__(self, weight_tensor):
        OpPruning.__init__(self)
        self.weight_tensor = weight_tensor

    def prune_input_channel(self, keep_indices):
        return [self.prune_tensor(self.weight_tensor, keep_indices, axis=2)]

    def prune_output_channel(self, keep_indices):
        return [self.prune_tensor(self.weight_tensor, keep_indices, axis=3)]


class SeparableConv2D(OpPruning):
    def __init__(self, depthwise_weight_tensor, pointwise_weight_tensor):
        OpPruning.__init__(self)
        self.depthwise_weight_tensor = depthwise_weight_tensor
        self.pointwise_weight_tensor = pointwise_weight_tensor

    def prune_input_channel(self, keep_indices):
        return [self.prune_tensor(self.depthwise_weight_tensor, keep_indices, axis=2),
                self.prune_tensor(self.pointwise_weight_tensor, keep_indices, axis=2)]

    def prune_output_channel(self, keep_indices):
        return [self.prune_tensor(self.pointwise_weight_tensor, keep_indices, axis=3)]


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
        ops = [self.prune_tensor(self.beta, keep_indices, axis=0),
               self.prune_tensor(self.moving_mean, keep_indices, axis=0),
               self.prune_tensor(self.moving_variance, keep_indices, axis=0)]
        if self.gamma:
            ops.append(self.prune_tensor(self.gamma, keep_indices, axis=0))
        return ops

    def prune_output_channel(self, keep_indices):
        return []


class BiasAdd(OpPruning):
    def __init__(self, bias):
        OpPruning.__init__(self)
        self.bias = bias

    def prune_input_channel(self, keep_indices):
        return []

    def prune_output_channel(self, keep_indices):
        return [self.prune_tensor(self.bias, keep_indices, axis=0)]


class Matmul(OpPruning):
    def __init__(self, weight_tensor):
        OpPruning.__init__(self)
        self.weight_tensor = weight_tensor

    def prune_input_channel(self, keep_indices):
        return [self.prune_tensor(self.weight_tensor, keep_indices, axis=0)]

    def prune_output_channel(self, keep_indices):
        return [self.prune_tensor(self.weight_tensor, keep_indices, axis=1)]


class ResidualConnection(OpPruning):
    """
     Residual connection consisting of (previous_residual_output + residual_output) 
     Hard to implement, skipping for now.
    """

    def prune_output_channel(self, keep_indices):
        pass

    def prune_input_channel(self, keep_indices):
        pass


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
        return [self.prune_tensor(self.weight_tensor, keep_indices, axis=3)]

    def prune_output_channel(self, keep_indices):
        return [self.prune_tensor(self.weight_tensor, keep_indices, axis=2)]
