from model import SeparableResnet
import tensorflow as tf

CHECKPOINT_FOLDER = "checkpoints"
with tf.Session() as session:
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))
    graph = SeparableResnet(input_tensor=input_placeholder)

    checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
    if checkpoint:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint)
    else:
        session.run(tf.variables_initializer(tf.global_variables()))

    graph_def = tf.graph_util.convert_variables_to_constants(
        session, session.graph_def, [graph.logits.op.name])

    print(graph.logits.op.name)
    print(input_placeholder.name)

    with tf.gfile.GFile("separable_resnet.pb", "wb") as f:
        f.write(graph_def.SerializeToString())
    f.close()
    session.close()
