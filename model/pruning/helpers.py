import tensorflow as tf


def sorted_union(a, b):
    merged = tf.concat([a, b], axis=0)
    unique_set = tf.unique(merged)
    reversed_set = unique_set[0] * -1
    reverse_sorted_set = tf.nn.top_k(reversed_set, k=tf.shape(reversed_set)[0])
    return tf.cast(tf.multiply(reverse_sorted_set.values, -1), dtype=tf.int32,
                   name="union/" + a.name.split(":")[0] + '/and/' + b.name.split(":")[0])


def assign(a, b):
    return tf.assign(a, b, validate_shape=False, name="assign/" + b.name.split(":")[0] + '/to/' + a.name.split(":")[0])
