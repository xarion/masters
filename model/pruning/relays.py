from model.pruning import Relay


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

    def start(self):
        self.prune_and_relay_next(None)


class LastNode(Relay):
    def __init__(self):
        Relay.__init__(self)
        self.relayed_backward = False

    def prune_and_relay_previous(self, keep_indices):
        if not self.relayed_backward:
            self.relayed_backward = True
            self.next_op.prune_and_relay_previous(None)
        else:
            raise Exception("Previous pruner is not set.")

    def set_next_op(self, previous_op):
        raise Exception("There can be nothing after the LastNode.")

    def prune_and_relay_next(self, keep_indices):
        self.prune_and_relay_previous(None)

    def set_previous_op(self, next_op):
        Relay.set_previous_op(self, next_op)
