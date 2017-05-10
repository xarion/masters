import numpy as np
import tensorflow as tf


class Prune:
    def __init__(self, graph_meta=None):
        self.graph_meta = graph_meta

    #  keep_indices must be 1 dimensional
    def prune_tensor(self, tensor, keep_indices, axis):
        partition_indices = np.zeros(tensor.get_shape())
        partition_indices = partition_indices.swapaxes(axis, 0)
        partition_indices[keep_indices, ...] = 1
        partition_indices = partition_indices.swapaxes(0, axis)

        partitions = tf.dynamic_partition(tensor, partition_indices, num_partitions=2)
        keep_partition = partitions[1]
        new_shape = tf.unstack(tensor.get_shape())
        new_shape[axis] = keep_indices.shape[0]
        new_tensor = tf.reshape(keep_partition, new_shape)
        if self.graph_meta is not None:
            self.graph_meta.set_variable_shape(tensor.name, new_shape)
        return tf.assign(tensor, new_tensor, validate_shape=False)
