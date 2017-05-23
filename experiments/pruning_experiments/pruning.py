#  Frozen TensorFlow graph as input
#  Pruned TensorFlow graph as output
#  Pruning based on weight values

import numpy as np
import tensorflow as tf

from graph_helper import extract_value, read_graph, write_graph, copy_const_with_value, \
    copy_operation_to_graph


def visit_nodes(graph):
    new_graph = tf.Graph()
    for operation in graph.get_operations():
        if operation.type == "Const":
            values = extract_value(operation)
            pruned_values = prune(values)
            copy_const_with_value(new_graph, operation, pruned_values)
        else:
            copy_operation_to_graph(new_graph, operation)
    return new_graph


def prune(values):
    values[np.abs(values) < 4e-1] = 0
    return values


graph = read_graph("models/pruned_output.pb")
with tf.Session() as sess:
    pruned_graph = visit_nodes(graph)
    write_graph(pruned_graph, ["pred"], sess, "models/pruned_output.pb")
