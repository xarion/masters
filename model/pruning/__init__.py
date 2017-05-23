import tensorflow as tf


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
    PRUNE_OP_COLLECTION  = 'prune_ops'
    PRUNABLES_COLLECTION = 'prunable_tensors'
    SHAPES_COLLECTION = 'new_tensor_shapes'

    def __init__(self):
        Relay.__init__(self)

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
        :param keep_indices: 1-D array of indices to be removed from axis 
        :param axis: axis that we will perform the pruning on
        :return: The assignment operation that should be run.
        """
        partition_indices = tf.zeros(tf.unstack(tensor.get_shape()))
        rank = tf.rank(tensor)
        swap_indices = tf.unstack(tf.range(0, rank))
        swap_indices[0] = axis
        swap_indices[axis] = 0
        swap_indices = tf.stack(swap_indices)
        partition_indices = tf.transpose(partition_indices, swap_indices)
        ones = tf.ones(tf.ones(tf.unstack(partition_indices.get_shape())[1:]))
        partition_indices = tf.scatter_update(partition_indices, keep_indices, ones)
        partition_indices = tf.transpose(partition_indices, swap_indices)

        partitions = tf.dynamic_partition(tensor, partition_indices, num_partitions=2)
        keep_partition = partitions[1]
        new_shape = tf.unstack(tensor.get_shape())
        new_shape[axis] = keep_indices.shape[0]
        new_tensor = tf.reshape(keep_partition, new_shape)

        assign_op = tf.assign(tensor, new_tensor, validate_shape=False)

        tf.add_to_collection(OpPruning.PRUNE_OP_COLLECTION, assign_op)
        tf.add_to_collection(OpPruning.PRUNABLES_COLLECTION, tensor)

        return assign_op

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
        self.register_stat_collection_ops()

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
        keep_indices = self.get_dimensions_to_keep()
        self.next_op.prune_and_relay_next(keep_indices)

    def prune_and_relay_previous(self, keep_indices):
        keep_indices = self.get_dimensions_to_keep()
        self.previous_op.prune_and_relay_previous(keep_indices)

    def register_stat_collection_ops(self):
        for stat_op in self.collect_stat_ops():
            tf.add_to_collection(OpStats.STAT_OP_COLLECTION, stat_op)
