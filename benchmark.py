import time

import numpy as np
import tensorflow as tf

from graph_helper import read_graph

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("g", "", """TensorFlow 'GraphDef' file to load.""")  # shorthand for graph_file
flags.DEFINE_string("o", "", """Output tensor name""")  # shorthand for output_tensor_name
flags.DEFINE_string("i", "", """Input tensor name""")  # shorthand for input_tensor_name
flags.DEFINE_integer("n", 1000, """How many times to run the inference""")  # shorthand for number_of_trials


# python benchmark.py --g="models/pruned_output.pb" --i="input:0" --o="pred:0" --n=10000


def main(unused_args):
    graph_file = FLAGS.g
    input_tensor_name = FLAGS.i
    output_tensor_name = FLAGS.o
    number_of_trials = FLAGS.n
    graph = read_graph(graph_file)
    with tf.Session(graph=graph) as sess:
        input_tensor = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)
        input_shape = input_tensor.get_shape()
        started_at = time.time()
        for i in xrange(1, number_of_trials):
            random_input = np.random.random_sample(input_shape)
            sess.run([output_tensor], feed_dict={input_tensor: random_input})
        ended_at = time.time()
        print "Execution took %.2f seconds" % (ended_at - started_at)


if __name__ == "__main__":
    tf.app.run()
