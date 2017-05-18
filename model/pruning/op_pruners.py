import tensorflow as tf

from . import OpPruning, Relay


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
        self.prune_tensor(self.beta, keep_indices, axis=0)
        self.prune_tensor(self.moving_mean, keep_indices, axis=0)
        self.prune_tensor(self.moving_variance, keep_indices, axis=0)
        if self.gamma:
            self.prune_tensor(self.gamma, keep_indices, axis=0)

    def prune_output_channel(self, keep_indices):
        pass


class BiasAdd(OpPruning):
    def __init__(self, bias):
        OpPruning.__init__(self)
        self.bias = bias

    def prune_input_channel(self, keep_indices):
        pass

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


class Branch(Relay):
    def __init__(self):
        Relay.__init__(self)
        self.joined = False
        self.other_next_op = None

    def set_next_op(self, next_op):
        if self.other_next_op is None:
            self.other_next_op = next_op
        # overrides the next_op when called a second time
        Relay.set_next_op(self, next_op)

    def prune_and_relay_next(self, keep_indices):
        self.other_next_op.prune_and_relay_next(keep_indices)
        Relay.prune_and_relay_next(keep_indices)

    def prune_and_relay_previous(self, keep_indices):
        if self.joined:
            Relay.prune_and_relay_previous(keep_indices)
        else:
            self.joined = True


class Join(Relay):
    def __init__(self):
        Relay.__init__(self)
        self.joined = False
        self.other_previous_op = None

    def set_previous_op(self, previous_op):
        if self.other_previous_op is None:
            self.other_previous_op = previous_op
        # overrides the previous_op when called a second time
        Relay.set_previous_op(self, previous_op)

    def prune_and_relay_next(self, keep_indices):
        if self.joined:
            Relay.prune_and_relay_next(keep_indices)
        else:
            self.joined = True

    def prune_and_relay_previous(self, keep_indices):
        self.other_previous_op.prune_and_relay_previous(keep_indices)
        Relay.prune_and_relay_previous(keep_indices)


class HeadNode(Relay):
    def __init__(self):
        Relay.__init__(self)
        self.relayed_forward = False

    def prune_and_relay_previous(self, keep_indices):
        self.prune_and_relay_next(None)

    def set_previous_op(self, previous_op):
        raise Exception("There can be nothing before Head.")

    def prune_and_relay_next(self, keep_indices):
        if not self.relayed_forward:
            self.relayed_forward = True
            self.next_op.prune_and_relay_next(None)

    def set_next_op(self, next_op):
        Relay.set_next_op(self, next_op)


class LastNode(Relay):
    def __init__(self):
        Relay.__init__(self)
        self.relayed_backward = False

    def prune_and_relay_previous(self, keep_indices):
        if not self.relayed_backward:
            self.relayed_backward = True
            self.next_op.prune_and_relay_previous(None)

    def set_next_op(self, previous_op):
        raise Exception("There can be nothing before Head.")

    def prune_and_relay_next(self, keep_indices):
        self.prune_and_relay_previous(None)

    def set_previous_op(self, next_op):
        Relay.set_previous_op(self, next_op)
