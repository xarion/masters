import tensorflow as tf

from model.pruning import OpStats
from model.pruning.helpers import assign


class ActivationValueStats(OpStats):
    def __init__(self, op):
        self.stat_op = None
        OpStats.__init__(self, op)

    def collect_stat_ops(self):
        with tf.variable_scope("stats"):
            stat_tensor = self.get_stat_tensor()
            axes_to_reduce = range(0, len(tf.unstack(self.op.get_shape())) - 1)
            current_means, current_variances = tf.nn.moments(self.op, axes=axes_to_reduce)

            previous_means = stat_tensor[0]
            previous_variances = stat_tensor[1]
            previous_counts = stat_tensor[2]
            new_counts = previous_counts + 1

            new_means = (previous_means * previous_counts + current_means) / new_counts
            new_variances = (((previous_variances + tf.square(current_means - new_means)) * previous_counts
                              + (current_variances + tf.square(previous_means - new_means)))
                             / new_counts)

            new_stat_tensor = tf.convert_to_tensor([new_means, new_variances, new_counts])
            stat_tensor_assign_op = assign(stat_tensor, new_stat_tensor)
        return [stat_tensor_assign_op]

    def get_stat_tensor(self):
        if not self.stat_op:
            self.stat_op = tf.get_variable(self.op.name.split(':')[0],
                                           shape=[3, self.op.get_shape()[-1]],
                                           initializer=tf.zeros_initializer())
        return self.stat_op

    def get_dimensions_to_keep(self):
        with tf.variable_scope("stats"):
            stat_tensor = self.get_stat_tensor()
            means = stat_tensor[0]
            variances = stat_tensor[1]

            return tf.cast(tf.squeeze(tf.where(tf.logical_and(means * 0.5 < variances, means * 1.5 > variances))),
                           dtype=tf.int32,
                           name="dimensions_to_keep")


class ActivationCountStats(OpStats):
    def __init__(self, op):
        self.stat_op = None
        OpStats.__init__(self, op)

    def collect_stat_ops(self):
        with tf.variable_scope("stats", [self.op]):
            stat_tensor = self.get_stat_tensor()
            axes_to_reduce = range(0, len(tf.unstack(self.op.get_shape())) - 1)
            stat = tf.cast((self.op > 0), tf.float32)
            new_stat_tensor = tf.add(tf.reduce_sum(stat, axis=axes_to_reduce), stat_tensor, name="update_stats")
            stat_tensor_assign_op = assign(stat_tensor, new_stat_tensor)
        return [stat_tensor_assign_op]

    def get_stat_tensor(self):
        if not self.stat_op:
            self.stat_op = tf.get_variable(self.op.name.split(':')[0],
                                           shape=self.op.get_shape()[-1],
                                           initializer=tf.zeros_initializer())

        return self.stat_op

    def get_dimensions_to_keep(self):
        stat_tensor = self.get_stat_tensor()
        with tf.variable_scope("keep_dims", [stat_tensor, self.op]):
            keep_dims = tf.cast(tf.squeeze(tf.where(stat_tensor > 0)), tf.int32, name="dimensions_to_keep")
        return keep_dims


class ActivationCorrelationStats(OpStats):
    def __init__(self, op):
        self.stat_op = None
        OpStats.__init__(self, op)

    def collect_stat_ops(self):
        with tf.variable_scope("stats"):
            stat_tensor = self.get_stat_tensor()
            flattened = tf.reshape(((tf.cast(self.op > 0, tf.float32) - 0.5) * 2), [-1, tf.shape(self.op)[-1]])
            values = tf.matmul(flattened, flattened, transpose_a=True)

            stat_tensor_assign_op = assign(stat_tensor, tf.add(values, stat_tensor, name="update_stats"))
        return [stat_tensor_assign_op]

    def get_stat_tensor(self):
        """
        :rtype: The tensor which holds the statistics that are required by this stat op
        """
        if self.stat_op is None:
            with tf.variable_scope("stats"):
                self.stat_op = tf.get_variable(self.op.name.split(':')[0],
                                               shape=[self.op.get_shape()[-1]] * 2,
                                               initializer=tf.zeros_initializer())
        return self.stat_op

    def get_dimensions_to_keep(self):

        stat_tensor = self.get_stat_tensor()
        if self.keep_indices is None:
            self.keep_indices = tf.Variable(validate_shape=False, name="keep_indices")

        no_diag_correlation_matrix = tf.matrix_set_diag(stat_tensor, tf.zeros(stat_tensor.get_shape()[-1]))

        non_uniqueness_matrix = tf.reduce_sum(no_diag_correlation_matrix, axis=1)
        (count, sum, sum_sqr, shift) = tf.nn.sufficient_statistics(non_uniqueness_matrix, axes=[0])
        stdev = tf.sqrt((sum_sqr - (sum * sum) / count) / (count - 1))
        mean = sum / count
        higher_threshold = mean + stdev
        higher_threshold = tf.cond(higher_threshold < 0,
                                   lambda: tf.constant(0, dtype=tf.float32),
                                   lambda: higher_threshold)
        raw_keep_dims = tf.where(non_uniqueness_matrix <= higher_threshold)

        return tf.cast(tf.squeeze(raw_keep_dims), dtype=tf.int32, name="dimensions_to_keep")
