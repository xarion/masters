import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from experiments.pruning_experiments.mnist_autoencoder.autoencoder import Autoencoder
from model.graph_meta import GraphMeta
from model.pruning import OpStats, OpPruning

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'Size of each training batch')
flags.DEFINE_integer('epochs', 2, 'Epochs to train for')

BATCH_SIZE = FLAGS.batch_size
EPOCHS = FLAGS.epochs
# Preprocessing Flags (only affect training data, not validation data)
CHECKPOINT_FOLDER = "checkpoints"

CHECKPOINT_NAME = "mnist-autoencoder"

DATA_DIR = "./data"
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


class Pruner:
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        self.graph = tf.Graph()
        self.latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
        self.graph_meta = GraphMeta(self.latest_checkpoint)

        with self.graph.as_default():
            self.model = Autoencoder(graph_meta=self.graph_meta, training=True, batch_size=BATCH_SIZE)

        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True),
                                  graph=self.graph)

    def prune(self):
        with self.graph.as_default():
            variables = tf.global_variables()
            variables.extend(tf.local_variables())
            saver = tf.train.Saver(var_list=variables, max_to_keep=5)
            self.model.head_pruner_node.start()
            self.session.run(tf.variables_initializer(tf.global_variables()))
            self.session.run(tf.variables_initializer(tf.local_variables()))

            latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
            self.session.run(tf.variables_initializer(tf.local_variables()))

            if latest_checkpoint:
                self.log("loading from checkpoint file: " + latest_checkpoint)
                saver.restore(self.session, latest_checkpoint)
            else:
                self.log("checkpoint not found, initializing variables.")
                self.session.run(tf.variables_initializer(tf.global_variables()))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
            stat_ops_collection = tf.get_collection(OpStats.STAT_OP_COLLECTION)
            prune_ops_collection = tf.get_collection(OpPruning.PRUNE_OP_COLLECTION)
            shape_ops_collection = tf.get_collection(OpPruning.SHAPES_COLLECTION)
            try:
                while mnist.train.epochs_completed < 1:
                    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                    _, = self.session.run([stat_ops_collection],
                                          feed_dict={self.model.do_validate: False,
                                                     self.model.input: batch_xs})
                _, new_shapes, step, = self.session.run(
                    [prune_ops_collection, shape_ops_collection, self.model.global_step])

                for new_shape in new_shapes:
                    tensor_name = new_shape.items()[0][0]
                    new_shape = new_shape.items()[0][1]
                    self.graph_meta.set_variable_shape(tensor_name, new_shape)
                self.graph_meta.save(self.graph_meta.step + 1)

                for key in self.graph_meta.dict.keys():
                    self.log(str(key) + str(self.graph_meta.dict[key]))
                saver.save(self.session, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME,
                           global_step=self.graph_meta.step + 1)

            except tf.errors.OutOfRangeError:
                self.log('Done training -- epoch limit reached')
            finally:
                saver.save(self.session, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=step)
                coord.request_stop()

            coord.join(threads)
            self.session.close()

    @staticmethod
    def log(message):
        sys.stdout.write(message + "\n")
        sys.stdout.flush()
        pass


def main(_):
    t = Pruner()
    t.prune()


if __name__ == '__main__':
    tf.app.run()
