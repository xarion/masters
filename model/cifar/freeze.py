import tensorflow as tf

from model.graph_meta import GraphMeta
from separable_resnet import SeparableResnet

CHECKPOINT_FOLDER = "checkpoints"


with tf.Session() as session:
    checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
    graph_meta = GraphMeta(checkpoint)
    graph = SeparableResnet(training=False, graph_meta=graph_meta)
    if checkpoint:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint)
        print('loading ' + checkpoint)
    else:
        raise Exception('not found')

    graph_def = tf.graph_util.convert_variables_to_constants(
        session, session.graph_def, [graph.logits.op.name])

    print(graph.logits.op.name)
    print(graph.input.name)

    with tf.gfile.GFile("separable_resnet-cifar-10.pb", "wb") as f:
        f.write(graph_def.SerializeToString())
    f.close()
    session.close()
