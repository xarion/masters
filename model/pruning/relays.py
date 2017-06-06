import tensorflow as tf

from model.pruning import Relay, OpStats
from model.pruning.helpers import sorted_union, assign


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
        self.previous_keep_indices = None
        self.other_keep_indices = None

    def set_next_op(self, next_op):
        if self.other_next_op is None:
            self.other_next_op = next_op
        # overrides the next_op when called a second time
        Relay.set_next_op(self, next_op)

    def prune_and_relay_next(self, keep_indices):
        self.previous_keep_indices = keep_indices
        self.other_next_op.prune_and_relay_next(keep_indices)
        self.next_op.prune_and_relay_next(keep_indices)

    def prune_and_relay_previous(self, keep_indices):
        if not self.joined:
            self.joined = True
            self.other_keep_indices = keep_indices
        else:
            union_op = sorted_union(self.previous_keep_indices, sorted_union(self.other_keep_indices, keep_indices))
            assign_keep_indices = assign(keep_indices, union_op)
            assign_other_keep_indices = assign(self.other_keep_indices, union_op)
            tf.add_to_collection(OpStats.STAT_OP_COLLECTION, assign_keep_indices)
            tf.add_to_collection(OpStats.STAT_OP_COLLECTION, assign_other_keep_indices)
            self.previous_op.prune_and_relay_previous(assign_keep_indices)


class ResidualJoin(Relay):
    """
    Joins the contents of connected residuals
    """

    def __init__(self):
        Relay.__init__(self)
        self.other_previous_op = None
        self.joined = False
        self.keep_indices = None
        self.other_keep_indices = None
        self.union_op = None

    def set_previous_op(self, previous_op):
        if self.other_previous_op is None:
            self.other_previous_op = previous_op
        # overrides the previous_op when called a second time
        Relay.set_previous_op(self, previous_op)

    def prune_and_relay_next(self, keep_indices):
        if not self.joined:
            self.joined = True
            self.keep_indices = keep_indices
        else:
            self.other_keep_indices = keep_indices
            self.union_op = sorted_union(self.keep_indices, self.other_keep_indices)
            self.next_op.prune_and_relay_next(self.keep_indices)

    def prune_and_relay_previous(self, keep_indices):
        self.union_op = sorted_union(keep_indices, self.union_op)
        assign_keep_indices = assign(self.keep_indices, self.union_op)
        assign_other_keep_indices = assign(self.other_keep_indices, self.union_op)
        tf.add_to_collection(OpStats.STAT_OP_COLLECTION, assign_keep_indices)
        tf.add_to_collection(OpStats.STAT_OP_COLLECTION, assign_other_keep_indices)
        self.other_previous_op.prune_and_relay_previous(assign_keep_indices)
        self.previous_op.prune_and_relay_previous(assign_keep_indices)


class EmptyForwardRelay(Relay):
    """
    Blocks keep_indices from relaying to next pruner
    """

    def __init__(self):
        Relay.__init__(self)

    def prune_and_relay_next(self, keep_indices):
        self.keep_indices = tf.Variable([], dtype=tf.int32, name="empty_relay")
        empty = assign(self.keep_indices, self.keep_indices)
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
        self.keep_indices = tf.Variable([], dtype=tf.int32, name="empty_relay")
        empty = assign(self.keep_indices, self.keep_indices)
        self.previous_op.prune_and_relay_previous(empty)
