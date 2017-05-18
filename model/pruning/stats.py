import tensorflow as tf

from . import OpStats


class ActivationValueStats(OpStats):
    def collect_stat_ops(self):
        with tf.variable_scope("stats"):
            stat_tensor = self.get_stat_tensor()
            axises_to_reduce = range(0, len(tf.unstack(self.op.get_shape())) - 1)
            stat_tensor_assign_op = tf.assign(stat_tensor,
                                              tf.reduce_sum(self.op, axis=axises_to_reduce) + stat_tensor)
        return [stat_tensor_assign_op]

    def get_stat_tensor(self):
        return tf.get_variable(self.op.name,
                               shape=self.op.get_shape()[-1],
                               initializer=tf.zeros_initializer())

    def get_dimensions_to_keep(self):
        with tf.variable_scope("stats", reuse=True):
            stat_tensor = self.get_stat_tensor()
            return tf.squeeze(tf.where(stat_tensor > 0))


class ActivationCountStats(OpStats):
    def collect_stat_ops(self):
        with tf.variable_scope("stats"):
            stat_tensor = self.get_stat_tensor()
            axises_to_reduce = range(0, len(tf.unstack(self.op.get_shape())) - 1)
            stat_tensor_assign_op = tf.assign(stat_tensor,
                                              tf.reduce_sum(self.op > 0, axis=axises_to_reduce) + stat_tensor)
        return [stat_tensor_assign_op]

    def get_stat_tensor(self):
        return tf.get_variable(self.op.name,
                               shape=self.op.get_shape(),
                               initializer=tf.zeros_initializer())

    def get_dimensions_to_keep(self):
        with tf.variable_scope("stats", reuse=True):
            stat_tensor = self.get_stat_tensor()
            return tf.squeeze(tf.where(stat_tensor > 0))
