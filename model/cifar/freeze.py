import tensorflow as tf

from separable_resnet import SeparableResnet

CHECKPOINT_FOLDER = "checkpoints"
with tf.Session() as session:
    # input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3))
    graph = SeparableResnet(training=False)

    checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
    saver = tf.train.Saver()
    if checkpoint:
        saver.restore(session, checkpoint)
    else:
        session.run(tf.variables_initializer(tf.global_variables()))
    saver.save(session, CHECKPOINT_FOLDER+ "/separable_resnet-cifar-10-resaved", global_step=0)
    session.close()

with tf.Session() as session:
    checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)

    if checkpoint:
        saver.restore(session, checkpoint)
    else:
        session.run(tf.variables_initializer(tf.global_variables()))

    graph_def = tf.graph_util.convert_variables_to_constants(
        session, session.graph_def, [graph.logits.op.name])

    print(graph.logits.op.name)
    print(graph.input.name)

    with tf.gfile.GFile("separable_resnet-cifar-10.pb", "wb") as f:
        f.write(graph_def.SerializeToString())
    f.close()
    session.close()
