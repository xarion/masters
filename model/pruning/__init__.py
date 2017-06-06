import tensorflow as tf

from model.pruning.helpers import sorted_union, assign


class Relay:
    def __init__(self):
        self.previous_op = None
        self.next_op = None

    def set_previous_op(self, previous_op):
        self.previous_op = previous_op
        previous_op.set_next_op(self)

    def set_next_op(self, next_op):
        self.next_op = next_op

    def prune_and_relay_next(self, keep_indices):
        raise NotImplementedError("Please Implement this method")

    def prune_and_relay_previous(self, keep_indices):
        raise NotImplementedError("Please Implement this method")


class OpPruning(Relay):
    PRUNE_OP_COLLECTION = 'prune_ops'
    SHAPES_COLLECTION = 'new_tensor_shapes'

    def __init__(self):
        Relay.__init__(self)
        self.related_tensors = {}

    def prune_input_channel(self, keep_indices):
        """
        :param keep_indices: The indices to be pruned from the input channel,
         must be flat. (an array of scalars) 
        :return: the assignment op that prunes the weights
        """
        raise NotImplementedError("Please Implement this method")

    def prune_output_channel(self, keep_indices):
        """
        :param keep_indices: The indices to be pruned from the output channel,
         must be flat. (an array of scalars) 
        :return: the assignment op that prunes the weights
        """
        raise NotImplementedError("Please Implement this method")

    def prune_tensor(self, tensor, keep_indices, axis):
        """
        :param tensor: the tensor to be pruned
        :param keep_indices: 1-D array of indices to be kept
        :param axis: axis that we will perform the pruning on
        :return: The assignment operation that should be run.
        """

        rank = tf.rank(tensor)
        swap_indices = tf.unstack(tf.range(0, rank))
        swap_indices[0] = axis
        swap_indices[axis] = 0
        swap_indices = tf.stack(swap_indices)
        swapped_tensor = tf.transpose(tensor, swap_indices)
        swapped_tensor = tf.cond(tf.reduce_max(keep_indices) > tf.shape(tensor)[axis],
                                 lambda: tf.Print(swapped_tensor,
                                                  [tensor.name, tf.shape(tensor), tf.reduce_max(keep_indices), axis],
                                                  summarize=4),
                                 lambda: swapped_tensor)
        pruned_swapped_tensor = tf.gather(swapped_tensor, keep_indices)
        new_tensor = tf.transpose(pruned_swapped_tensor, swap_indices)
        assign_op = tf.assign(tensor, new_tensor, validate_shape=False, name="assign_pruned/" + tensor.name.split(':')[0])

        tf.add_to_collection(OpPruning.PRUNE_OP_COLLECTION, assign_op)

        return assign_op

    def prune(self, tensor, keep_indices, axis):
        new_tensor = self.prune_tensor(tensor, keep_indices, axis)
        new_ops = []
        for related_tensor in self.get_related_tensors(tensor):
            new_ops.append(self.prune_tensor(related_tensor, keep_indices, axis))

        tf.add_to_collection(self.SHAPES_COLLECTION,
                             {self.get_underlying_tensor(new_tensor).name: tf.shape(new_tensor)})
        self.set_related_tensors(new_tensor, new_ops)
        return new_tensor

    def get_related_tensors(self, tensor):
        if tensor not in self.related_tensors:
            adam_tensor_name_part = "/" + tensor.name.split(':')[0] + '/Adam'
            ops = []
            for other_tensor in tf.global_variables():
                if other_tensor.name.find(adam_tensor_name_part) >= 0:
                    if other_tensor.get_shape() == tensor.get_shape():
                        ops.append(other_tensor)
            self.related_tensors[tensor] = ops

        return self.related_tensors[tensor]

    @staticmethod
    def get_underlying_tensor(assign_op):
        t = assign_op
        while t.op.type == "Assign":
            t = t.op.inputs[0]
        return t

    def set_related_tensors(self, tensor, related_tensors):
        self.related_tensors[tensor] = related_tensors

    def prune_and_relay_next(self, keep_indices):
        if keep_indices is not None:
            self.prune_input_channel(keep_indices)
        self.next_op.prune_and_relay_next(keep_indices)

    def prune_and_relay_previous(self, keep_indices):
        if keep_indices is not None:
            self.prune_output_channel(keep_indices)
        self.previous_op.prune_and_relay_previous(keep_indices)


class OpStats(Relay):
    STAT_OP_COLLECTION = 'collect_stat_ops'

    def __init__(self, op):
        Relay.__init__(self)
        self.op = op
        self.keep_indices = None

    def collect_stat_ops(self):
        """
        Abstract method definition for stat collection op. It a list of ops/tensors  
        :return: A set of ops to make stat collection. 
         Note: Returned ops do not correspond to stat tensors, see `get_stat_tensor` for those stats. 
        """
        raise NotImplementedError("Please Implement this method")

    def get_stat_tensor(self):
        """
        :return: returns the stat tensor 
        """
        raise NotImplementedError("Please Implement this method")

    def get_dimensions_to_keep(self):
        """
        :return: keep_dims for pruning ops 
        """
        raise NotImplementedError("Please Implement this method")

    def prune_and_relay_next(self, keep_indices):
        self.register_stat_collection_ops()
        if self.keep_indices is None:
            self.set_keep_indices()
        self.next_op.prune_and_relay_next(self.keep_indices)

    def prune_and_relay_previous(self, keep_indices):
        if self.keep_indices is None:
            self.set_keep_indices()
        self.previous_op.prune_and_relay_previous(self.keep_indices)

    def register_stat_collection_ops(self):
        for stat_op in self.collect_stat_ops():
            tf.add_to_collection(OpStats.STAT_OP_COLLECTION, stat_op)

    def set_keep_indices(self):
        if self.keep_indices is None:
            self.keep_indices = tf.Variable([], dtype=tf.int32, name="keep_indices")
        self.keep_indices = assign(self.keep_indices, sorted_union(self.keep_indices, self.get_dimensions_to_keep()))
