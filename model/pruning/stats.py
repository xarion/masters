import tensorflow as tf

from . import OpStats


class ActivationValueStats(OpStats):
    def collect_stat_ops(self):
        with tf.variable_scope("stats"):
            stat_tensor = self.get_stat_tensor()
            axes_to_reduce = range(0, len(tf.unstack(self.op.get_shape())) - 1)
            stat_tensor_assign_op = tf.assign(stat_tensor,
                                              tf.reduce_sum(self.op, axis=axes_to_reduce) + stat_tensor)
        return [stat_tensor_assign_op]

    def get_stat_tensor(self):
        return tf.get_variable(self.op.name.split(':')[0],
                               shape=self.op.get_shape()[-1],
                               initializer=tf.zeros_initializer())

    def get_dimensions_to_keep(self):
        with tf.variable_scope("stats", reuse=True):
            stat_tensor = self.get_stat_tensor()
            return tf.squeeze(tf.where(stat_tensor > 0))


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
        if self.stat_op:
            return self.stat_op
        else:
            return tf.get_variable(self.op.name.split(':')[0],
                                   shape=self.op.get_shape()[-1],
                                   initializer=tf.zeros_initializer())

    def get_dimensions_to_keep(self):
        stat_tensor = self.get_stat_tensor()
        return tf.squeeze(tf.where(stat_tensor > 0))
