import tensorflow as tf
from tensorflow.python.framework import tensor_util, graph_util


def read_graph(graph_file):
    if tf.gfile.Exists(graph_file):
        graph_def = tf.GraphDef()
        with open(graph_file, "rb") as f:
            graph_def.ParseFromString(f.read())
    else:
        raise Exception("Model file not found!")
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, input_map={}, name="")
    return graph


def extract_value(operation):
    return tensor_util.MakeNdarray(operation.get_attr("value"))


def write_graph(graph, output_layers, session, file_name):
    constant_graph = graph_util.convert_variables_to_constants(
        session, graph.as_graph_def(), output_layers)
    with tf.gfile.GFile(file_name, "wb") as f:
        f.write(constant_graph.SerializeToString())


def copy_const_with_value(graph, operation, values):
    tensor = tf.convert_to_tensor(values, name=operation.name)
    return copy_operation_to_graph(graph, tensor.op)


def copy_operation_to_graph(graph, operation):
    inputs = [graph.as_graph_element(input_object.op.name + ":0", allow_operation=False) for input_object in operation.inputs]
    output_dtypes = [output.dtype for output in operation.outputs]
    return graph.create_op(operation.type, inputs, output_dtypes, op_def=operation.op_def,
                           attrs=operation.node_def.attr, name=operation.name)
