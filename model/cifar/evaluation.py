from __future__ import print_function

import os
import sys
import tarfile

import numpy as np
import tensorflow as tf
from six.moves import urllib

import data as cifar10
from model.cifar.separable_resnet import SeparableResnet
from model.graph_meta import GraphMeta

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 200, 'Size of each training batch')


DATA_DIR = "./data"
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

max_sample_count = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
# max_sample_count = 33


class Evaluation:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.graph_meta = GraphMeta(self.checkpoint)

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.model = SeparableResnet(graph_meta=self.graph_meta,
                                         training=True, batch_size=FLAGS.batch_size)

        self.session = tf.Session(graph=self.graph)

    def evaluate(self):
        with self.graph.as_default():
            saver = tf.train.Saver(max_to_keep=1)

            self.session.run(tf.variables_initializer(tf.global_variables()))
            self.session.run(tf.variables_initializer(tf.local_variables()))

            if self.checkpoint:
                self.log("loading from checkpoint file: " + self.checkpoint)
                saver.restore(self.session, self.checkpoint)
            else:
                raise Exception("Checkpoint file not found. Can not proceed with pruning.")
            # Evaluation.log("loaded checkpoint file")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
            accuracies = []
            try:
                sample_count = 0
                while not coord.should_stop():
                    accuracy, = self.session.run([self.model.top_1_accuracy], feed_dict={self.model.do_validate: True})
                    accuracies.append(accuracy)
                    sample_count += FLAGS.batch_size
                    self.log("processed: %d/%d" % (sample_count, max_sample_count))
                    if sample_count >= max_sample_count:
                        break

                self.log("%s \t top-1 accuracy: %.4f" % (self.checkpoint, np.mean(np.array(accuracies))))

            except tf.errors.OutOfRangeError:
                self.log('Finished evaluating gracefully')
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
            Evaluation.log('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
        if not os.path.exists(extracted_dir_path):
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    checkpoints = [
        'SEP-RESNET-34-250001'
        # 'SEP-RESNET-52-160000',
        # 'SEP-RESNET-52-200000',
        # 'SEP-RESNET-52-60000',
                   ]

    for checkpoint in checkpoints:
        e = Evaluation("checkpoints/" + checkpoint)
        e.evaluate()


if __name__ == '__main__':
    tf.app.run()
