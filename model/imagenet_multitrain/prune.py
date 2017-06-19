from __future__ import print_function

import os
import sys
import tarfile

import tensorflow as tf
from six.moves import urllib

import data as cifar10
from model.cifar.separable_resnet import SeparableResnet
from model.graph_meta import GraphMeta
from model.pruning import OpStats, OpPruning

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, 'Size of each training batch')

CHECKPOINT_FOLDER = "checkpoints"
CHECKPOINT_STEP = 50000
CHECKPOINT_NAME = "SEP-RESNET-52"

DATA_DIR = "./data"
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

# max_sample_count = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


max_sample_count = 300


class Prune:
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        self.latest_checkpoint = "checkpoints/SEP-RESNET-52-80000"
        self.graph_meta = GraphMeta(self.latest_checkpoint)

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.model = SeparableResnet(graph_meta=self.graph_meta,
                                         training=True,
                                         batch_size=FLAGS.batch_size)

        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True),
                                  graph=self.graph)

    def prune(self):
        with self.graph.as_default():
            variables = tf.global_variables()
            variables.extend(tf.local_variables())
            saver = tf.train.Saver(var_list=variables, max_to_keep=1)
            self.model.head_pruner_node.start()
            self.session.run(tf.variables_initializer(tf.global_variables()))
            self.session.run(tf.variables_initializer(tf.local_variables()))

            if self.latest_checkpoint:
                self.log("loading from checkpoint file: " + self.latest_checkpoint)
                saver.restore(self.session, self.latest_checkpoint)
            else:
                raise Exception("Checkpoint file not found. Can not proceed with pruning.")
            Prune.log("loaded checkpoint file")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
            stat_ops_collection = tf.get_collection(OpStats.STAT_OP_COLLECTION)
            prune_ops_collection = tf.get_collection(OpPruning.PRUNE_OP_COLLECTION)
            shape_ops_collection = tf.get_collection(OpPruning.SHAPES_COLLECTION)

            self.session.run(tf.variables_initializer(tf.global_variables()))

            try:
                sample_count = 0
                while not coord.should_stop():
                    _, logits, = self.session.run([stat_ops_collection, self.model.logits],
                                                  feed_dict={self.model.do_validate: False})
                    sample_count += FLAGS.batch_size
                    self.log("processed: %d/%d" % (sample_count, max_sample_count))
                    if sample_count >= max_sample_count:
                        break
                _, new_shapes, = self.session.run([prune_ops_collection, shape_ops_collection])
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
                self.log('Done pruning')
            finally:
                coord.request_stop()

            coord.join(threads)
            self.session.close()

    @staticmethod
    def log(message):
        sys.stdout.write(message)
        sys.stdout.write("\n")
        sys.stdout.flush()
        pass

    @staticmethod
    def maybe_download_and_extract():
        """Download and extract the tarball from Alex's website."""
        dest_directory = "./data"
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                                 float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)

            statinfo = os.stat(filepath)
            Prune.log('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
        if not os.path.exists(extracted_dir_path):
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    p = Prune()
    p.prune()


if __name__ == '__main__':
    tf.app.run()
