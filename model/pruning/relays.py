import tensorflow as tf

from model.pruning import Relay, OpStats


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

    def start(self):
        self.prune_and_relay_next(None)


class LastNode(Relay):
    def __init__(self):
        Relay.__init__(self)
        self.relayed_backward = False

    def prune_and_relay_previous(self, keep_indices):
        if not self.relayed_backward:
            self.relayed_backward = True
            self.previous_op.prune_and_relay_previous(None)
        else:
            raise Exception("Already relayed backward once.")

    def set_next_op(self, previous_op):
        raise Exception("There can be nothing after the LastNode.")

    def prune_and_relay_next(self, keep_indices):
        self.prune_and_relay_previous(None)

    def set_previous_op(self, next_op):
        Relay.set_previous_op(self, next_op)


class ResidualBranch(Relay):
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
        self.next_op.prune_and_relay_next(keep_indices)

    def prune_and_relay_previous(self, keep_indices):
        if self.joined:
            self.previous_op.prune_and_relay_previous(keep_indices)
        else:
            self.joined = True


class ResidualConnectionWithStat(Relay):
    """
    Joins the contents of connected residuals
    """

    def __init__(self, residual_pruner, other_pruner):
        Relay.__init__(self)
        self.stat_op = None
        self.residual_pruner = residual_pruner
        self.residual_pruner.set_next_op(self)
        self.set_previous_op(other_pruner)
        self.joined = False
        self.keep_indices = None

    def set_stat_op(self, stat_op):
        """
        sets the stat op that determines the dimensions to be kept for this connection.
        :param stat_op:
        :return:
        """
        self.stat_op = stat_op

    def prune_and_relay_next(self, keep_indices):
        if self.keep_indices is None:
            self.keep_indices = tf.Variable([], dtype=tf.int32)

        if keep_indices is not None:
            assign_op = tf.assign(self.keep_indices,
                                  self.sorted_union(keep_indices, self.keep_indices),
                                  validate_shape=False)
            self.keep_indices = assign_op
            tf.add_to_collection(OpStats.STAT_OP_COLLECTION, assign_op)

        if not self.joined:
            self.joined = True
        else:
            stat_keep_indices = self.stat_op.get_dimensions_to_keep()
            assign_op = tf.assign(self.keep_indices,
                                  self.sorted_union(self.keep_indices, stat_keep_indices),
                                  validate_shape=False)
            self.keep_indices = assign_op
            tf.add_to_collection(OpStats.STAT_OP_COLLECTION, assign_op)
            self.next_op.prune_and_relay_next(self.keep_indices)

    def prune_and_relay_previous(self, keep_indices):
        self.previous_op.prune_and_relay_previous(self.keep_indices)
        self.residual_pruner.prune_and_relay_previous(self.keep_indices)

    @staticmethod
    def sorted_union(a, b):
        merged = tf.concat([a, b], axis=0)
        set = tf.unique(merged)
        reversed_set = set[0] * -1
        reverse_sorted_set = tf.nn.top_k(reversed_set, k=tf.shape(reversed_set)[0])
        return tf.cast(reverse_sorted_set.values * -1, dtype=tf.int32)


class EmptyForwardRelay(Relay):
    """
    Blocks keep_indices from relaying to next pruner
    """

    def __init__(self):
        Relay.__init__(self)

    def prune_and_relay_next(self, keep_indices):
        empty = tf.Variable([], dtype=tf.int32)
        empty = tf.assign(empty, empty)
        self.next_op.prune_and_relay_next(empty)

    def prune_and_relay_previous(self, keep_indices):
        self.previous_op.prune_and_relay_previous(keep_indices)


class EmptyBackwardRelay(Relay):
    """
    Blocks keep_indices from relaying to previous pruner
    """

    def __init__(self):
        Relay.__init__(self)

    def prune_and_relay_next(self, keep_indices):
        self.next_op.prune_and_relay_next(keep_indices)

    def prune_and_relay_previous(self, keep_indices):
        empty = tf.Variable([], dtype=tf.int32)
        empty = tf.assign(empty, empty)
        self.previous_op.prune_and_relay_previous(empty)
