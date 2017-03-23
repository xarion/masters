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


#  to reduce the number of parameters, this eq should hold true
#  r < (m * n / (m + n))
#  So here we choose r as (m * n / (m + n)) / K
def choose_rank(weights):
    m, n = weights.shape
    new_rank = int((m * n / (m + n)) / 2)
    return new_rank


#  Given a set of weights, calculates the SVD(s, u, v)
#  Uses choose_rank to calculate a lower rank
#  returns low rank s, u, v
def decompose_low_rank(weights):
    u, s, v = np.linalg.svd(weights, compute_uv=True, full_matrices=True)
    new_rank = choose_rank(weights)
    low_rank_u = u[:, :new_rank]
    low_rank_s = s[:new_rank]
    low_rank_v = v[:new_rank, :]
    return low_rank_u, low_rank_s, low_rank_v


#  there could be other compositions such as w1 = s dot sqrt(s) and w2 = sqrt(s) dot v.
#  output is always composed weights for the decomposition, to be multiplied with MatMul.
def create_svd_composition_constants(svd, op_name):
    u, s, v = svd
    us = tf.constant(np.dot(u, np.diag(s)), name=op_name + "/svd/u_dot_s")
    v = tf.constant(v, name=op_name + "/svd/v")
    return us, v


def create_low_rank_composition_layer(input_tensor, w1, w2, op_name):
    intermediate_tensor = tf.matmul(input_tensor, w1, name=op_name + "/svd/intermediate")
    output_tensor = tf.matmul(intermediate_tensor, w2, name=op_name + "/svd/output")
    return intermediate_tensor, output_tensor


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
        if operation.type == "MatMul":
            weight_tensor = None
            input_tensor = None
            for operation_input in operation.inputs:
                if operation_input.op.type == "Identity" or operation_input.op.type == "Const":
                    weight_tensor = operation_input.eval(session=session)
                else:
                    input_tensor = operation_input
            if any(filter(lambda s: s < 2, weight_tensor.shape)):
                new_operations.append(operation)
            else:
                svd = decompose_low_rank(weight_tensor)
                us, v = create_svd_composition_constants(svd, operation.name)
                intermediate_tensor, output_tensor = create_low_rank_composition_layer(input_tensor, us, v, operation.name)
                new_operations.append(us.op)
                new_operations.append(v.op)
                new_operations.append(intermediate_tensor.op)
                new_operations.append(output_tensor.op)
                replacements[operation.name] = output_tensor.op.name
        else:
            new_operations.append(operation)
        for op in new_operations:
            copy_operation_to_graph(new_graph, op, replacements)
    return new_graph


graph = read_graph("models/output.pb")
with tf.Session(graph=graph) as sess:
    pruned_graph = low_rank_graph_approximation(graph, sess)
    write_graph(pruned_graph, ["pred"], sess, "models/svd_output.pb")
