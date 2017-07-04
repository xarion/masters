#  Decompose MatMul Weights[M, N] using SVD as U[M, R], S[R,R], V[R, N]
#  where [U dot S dot V] = Weights
#  S represents the Singular Value Matrix, which is an ordered matrix.
#  Choose a Low Rank(K) that reduces the number of parameters, using a constant parameter K
#  Create Low Rank versions of U, S, V as U_low[M, K], S_low[K, K], V_low[K, N]
#  where [U_low dot S_low dot V_low] ~= Weights
#  Combine U_low and S_low to US_low
#  Compose a new layer replacing the original layer with low rank composition
#  which implies [Input dot Weights] ~= [Input dot US_low dot V_low]

import numpy as np
import tensorflow as tf


from graph_helper import copy_operation_to_graph, read_graph, write_graph

ERROR_THRESHOLD = 0.005

#  to reduce the number of parameters, this eq should hold true
#  r < (m * n / (m + n))
#  So here we choose r as (m * n / (m + n)) / K
def choose_rank(weights):
    original_weights = weights
    if len(weights.shape) > 2:
        shape = (np.prod(weights.shape[0:-1]), weights.shape[-1])
    elif len(weights.shape) == 2:
        shape = weights.shape
    else:
        raise Exception("kernel size unsupported")
    m, n = shape
    weights = np.reshape(weights, shape)
    rounded_weights = np.around(weights, decimals=2)
    U, s, V = np.linalg.svd(rounded_weights, full_matrices=False)
    error = 0
    r = s.shape[0]
    while (error < ERROR_THRESHOLD) and r > 1:
        s = s[0:-1]
        U = U[:, 0:-1]
        V = V[0:-1, :]
        S = np.diag(s)
        new_weights = np.dot(U, np.dot(S, V))
        error = np.mean(np.square(weights - new_weights))
        r -= 1

    if r < ((m * n) / (m + n)):
        print("%s -> %s %s %s" % (str(original_weights.shape), str(U.shape), str(s.shape), str(V.shape)))
        return U, s, V
    else:
        return original_weights


# there could be other compositions such as w1 = s dot sqrt(s) and w2 = sqrt(s) dot v.
#  output is always composed weights for the decomposition, to be multiplied with MatMul.
def create_svd_composition_constants(svd, op_name):
    u, s, v = svd
    us = np.dot(u, np.diag(s))
    return us, v


def create_matmul_low_rank_composition_layer(input_tensor, w1, w2, op_name):
    w1 = tf.constant(w1, name=op_name + "/svd/u_dot_s")
    w2 = tf.constant(w2, name=op_name + "/svd/v")
    intermediate_tensor = tf.matmul(input_tensor, w1, name=op_name + "/svd/intermediate")
    output_tensor = tf.matmul(intermediate_tensor, w2, name=op_name + "/svd/output")
    return w1, w2, intermediate_tensor, output_tensor


def create_conv2d_low_rank_composition_layer(input_tensor, w1, w2, previous_weight_shape, strides, padding, op_name):
    first_shape = list(previous_weight_shape[0:-1])
    first_shape.append(-1)
    w1 = np.reshape(w1, first_shape)

    second_shape = [1, 1, -1, previous_weight_shape[-1]]
    w2 = np.reshape(w2, second_shape)

    w1 = tf.constant(w1, name=op_name + "/svd/u_dot_s")
    w2 = tf.constant(w2, name=op_name + "/svd/v")
    intermediate_tensor = tf.nn.conv2d(input_tensor, filter=w1, strides=strides, padding=padding,
                                       name=op_name + "/svd/intermediate")
    output_tensor = tf.nn.conv2d(intermediate_tensor, filter=w2, strides=[1, 1, 1, 1], padding="SAME",
                                 name=op_name + "/svd/approximation")
    return w1, w2, intermediate_tensor, output_tensor


def low_rank_graph_approximation(graph, session):
    new_graph = tf.Graph()
    #  When we replace an operation X with operation Y,
    #  operations that take the `output tensor of X` as input should use the `output tensor of Y` instead.
    #  In those cases, we'll create an entry in the replacements dict as (X.name -> Y.name)
    replacements = {}
    #  We don't care about replaced weights,
    #  because when writing the graph, we're using `graph_util.convert_variables_to_constants`
    #  which rewrites the whole graph by ignoring the unused variables/constants
    for operation in graph.get_operations():
        new_operations = []
        if is_decomposable(operation.type):
            weight_tensor = None
            input_tensor = None
            for operation_input in operation.inputs:
                if is_weight_type(operation_input.op.type) and operation_input.name.find("Dropout") is -1:
                    try:
                        weight_tensor = operation_input.eval(session=session)
                    except Exception as e:
                        print operation_input.op.type
                        print operation_input.name
                        print e.message
                else:
                    input_tensor = operation_input
            print("choosing rank for %s" % operation.name)
            new_weights = choose_rank(weight_tensor)
            # new_weights = weight_tensor
            if type(new_weights) == tuple:
                us, v = create_svd_composition_constants(new_weights, operation.name)
                if is_matmul(operation.type):
                    us, v, intermediate_tensor, output_tensor = create_matmul_low_rank_composition_layer(input_tensor,
                                                                                                         us, v,
                                                                                                         operation.name)
                elif is_conv2d(operation.type):
                    padding = operation.node_def.attr['padding'].s
                    strides = list(operation.node_def.attr['strides'].list.i)

                    us, v, intermediate_tensor, output_tensor = create_conv2d_low_rank_composition_layer(input_tensor,
                                                                                                         us, v,
                                                                                                         weight_tensor.shape,
                                                                                                         strides,
                                                                                                         padding,
                                                                                                         operation.name)
                else:
                    raise Exception("How did i get here?")

                new_operations.append(us.op)
                new_operations.append(v.op)
                new_operations.append(intermediate_tensor.op)
                new_operations.append(output_tensor.op)
                replacements[operation.name] = output_tensor.op.name
            else:
                new_operations.append(operation)
        else:
            new_operations.append(operation)

        for op in new_operations:
            copy_operation_to_graph(new_graph, op, replacements)
    return new_graph


def is_matmul(operation_type):
    return operation_type == "MatMul"


def is_conv2d(operation_type):
    return operation_type == "Conv2D"


def is_decomposable(operation_type):
    return is_matmul(operation_type) or is_conv2d(operation_type)


def is_weight_type(operation_type):
    return operation_type == "Identity" or operation_type == "Const"


graph = read_graph("inception_resnet_v2-trained.pb")
with tf.Session(graph=graph) as sess:
    pruned_graph = low_rank_graph_approximation(graph, sess)
    write_graph(pruned_graph, ["InceptionResnetV2/Logits/Logits/BiasAdd"], sess,
                "inception_resnet_v2-new-threshold-" + str(ERROR_THRESHOLD) + ".pb")
