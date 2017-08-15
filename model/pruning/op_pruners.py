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
        #  mid_pruners are the operations between depthwise and pointwise convolutions
        self.mid_pruners = list()

    def prune_input_channel(self, keep_indices):
        self.depthwise_weight_tensor = self.prune(self.depthwise_weight_tensor, keep_indices, axis=2)
        self.pointwise_weight_tensor = self.prune(self.pointwise_weight_tensor, keep_indices, axis=2)
        for extension in self.mid_pruners:
            extension.prune_output_channel(keep_indices)

    def prune_output_channel(self, keep_indices):
        self.pointwise_weight_tensor = self.prune(self.pointwise_weight_tensor, keep_indices, axis=3)

    def extend(self, pruner):
        self.mid_pruners.append(pruner)



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


class FeaturePadding(OpPruning):
    def __init__(self, beginning_pad_count, ending_pad_count, feature_count):
        OpPruning.__init__(self)
        self.beginning_pad_count = beginning_pad_count
        self.ending_pad_count = ending_pad_count
        self.feature_count = feature_count

    def prune_input_channel(self, keep_indices):
        pass

    def prune_output_channel(self, keep_indices):
        kept_ending_indices = tf.where(keep_indices >= self.beginning_pad_count + self.feature_count)
        new_ending_pad_count = tf.reduce_sum(tf.cast(kept_ending_indices, tf.int32))

        kept_beginning_indices = tf.where(keep_indices < self.beginning_pad_count)
        new_beginning_pad_count = tf.reduce_sum(tf.cast(kept_beginning_indices, tf.int32))

        assign_ending_pad_count = tf.assign(self.ending_pad_count, new_ending_pad_count)
        assign_beginning_pad_count = tf.assign(self.beginning_pad_count, new_beginning_pad_count)

        tf.add_to_collection(OpPruning.PRUNE_OP_COLLECTION, assign_beginning_pad_count)
        tf.add_to_collection(OpPruning.PRUNE_OP_COLLECTION, assign_ending_pad_count)
