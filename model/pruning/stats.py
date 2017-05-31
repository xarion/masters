import tensorflow as tf

from model.pruning import OpStats


class ActivationValueStats(OpStats):
    def __init__(self, op):
        self.stat_op = None
        OpStats.__init__(self, op)

    def collect_stat_ops(self):
        with tf.variable_scope("stats"):
            stat_tensor = self.get_stat_tensor()
            axes_to_reduce = range(0, len(tf.unstack(self.op.get_shape())) - 1)
            stat_tensor_assign_op = tf.assign(stat_tensor,
                                              tf.reduce_sum(self.op, axis=axes_to_reduce) + stat_tensor)
        return [stat_tensor_assign_op]

    def get_stat_tensor(self):
        if not self.stat_op:
            self.stat_op = tf.get_variable(self.op.name.split(':')[0],
                                           shape=self.op.get_shape()[-1],
                                           initializer=tf.zeros_initializer())
        return self.stat_op

    def get_dimensions_to_keep(self):
        with tf.variable_scope("stats"):
            stat_tensor = self.get_stat_tensor()
            threshold = tf.reduce_mean(stat_tensor)
            return tf.cast(tf.squeeze(tf.where(stat_tensor >= threshold)), dtype=tf.int32)


class ActivationCountStats(OpStats):
    def __init__(self, op):
        self.stat_op = None
        OpStats.__init__(self, op)

    def collect_stat_ops(self):
        with tf.variable_scope("stats", [self.op]):
            stat_tensor = self.get_stat_tensor()
            axes_to_reduce = range(0, len(tf.unstack(self.op.get_shape())) - 1)
            stat = tf.cast((self.op > 0), tf.float32)
            stat_tensor_assign_op = tf.assign(stat_tensor,
                                              tf.reduce_sum(stat, axis=axes_to_reduce) + stat_tensor)
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
            keep_dims = tf.cast(tf.squeeze(tf.where(stat_tensor > 0)), tf.int32, name=self.op.name.split(':')[0])
        return keep_dims
