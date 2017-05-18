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

    @staticmethod
    def prune_tensor(tensor, keep_indices, axis):
        """    
        :param tensor: the tensor to be pruned 
        :param keep_indices: 1-D array of indices to be removed from axis 
        :param axis: axis that we will perform the pruning on
        :return: The assignment operation that should be run.
        """
        partition_indices = tf.zeros(tf.unstack(tensor.get_shape()))
        partition_indices = partition_indices.swapaxes(axis, 0)
        partition_indices[keep_indices, ...] = 1
        partition_indices = partition_indices.swapaxes(0, axis)

        partitions = tf.dynamic_partition(tensor, partition_indices, num_partitions=2)
        keep_partition = partitions[1]
        new_shape = tf.unstack(tensor.get_shape())
        new_shape[axis] = keep_indices.shape[0]
        new_tensor = tf.reshape(keep_partition, new_shape)
        # if self.graph_meta is not None:
        #     self.graph_meta.set_variable_shape(tensor.name, new_shape)
        return tf.assign(tensor, new_tensor, validate_shape=False)

    def prune_and_relay_next(self, keep_indices):
        if keep_indices is not None:
            self.prune_input_channel(keep_indices)
        self.next_op.prune_and_relay_next(keep_indices)

    def prune_and_relay_previous(self, keep_indices):
        if keep_indices is not None:
            self.prune_output_channel(keep_indices)
        self.previous_op.prune_and_relay_previous(keep_indices)


class OpStats(Relay):
    def __init__(self, op):
        Relay.__init__(self)
        self.op = op

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
