from model import SeparableResnet
import tensorflow as tf

with tf.Session() as session:
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))
    graph = SeparableResnet(input_placeholder)

    session.run(tf.variables_initializer(tf.global_variables()))
    graph_def = tf.graph_util.convert_variables_to_constants(
        session, session.graph_def, [graph.logits.op.name])

    print(graph.logits.op.name)
    print(input_placeholder.name)

    with tf.gfile.GFile("normal_resnet.pb", "wb") as f:
        f.write(graph_def.SerializeToString())
    f.close()
    session.close()
