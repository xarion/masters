import os
import sys
import tarfile

import tensorflow as tf
from six.moves import urllib

from experiments.structures.separable_convolutions.cifar10.cifar10 import Cifar10Model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'Size of each training batch')

# Preprocessing Flags (only affect training data, not validation data)
CHECKPOINT_FOLDER = "checkpoints_dec"
CHECKPOINT_STEP = 10000
CHECKPOINT_NAME = "kernel-composed-cifar"

DATA_DIR = "./data"
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


class Train:
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.maybe_download_and_extract()
            self.model = Cifar10Model(training=True, batch_size=FLAGS.batch_size)

        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True),
                                  graph=self.graph)

    def train(self):
        with self.graph.as_default():
            self.session.run(tf.variables_initializer(tf.local_variables()))
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("summaries/train_dec", self.graph)
            test_writer = tf.summary.FileWriter("summaries/test_dec", self.graph)
            saver = tf.train.Saver(max_to_keep=2)

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
            try:
                while not coord.should_stop():
                    m, _, loss, step, = self.session.run([merged,
                                                          self.model.train_step,
                                                          self.model.loss,
                                                          self.model.global_step],
                                                         feed_dict={self.model.do_validate: False})

                    train_writer.add_summary(m, step)
                    m, top1, = self.session.run([merged, self.model.top_1_accuracy],
                                                feed_dict={self.model.do_validate: True})
                    test_writer.add_summary(m, step)

                    if step % CHECKPOINT_STEP == 0:
                        saver.save(self.session, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=step)
            except tf.errors.OutOfRangeError:
                self.log('Done training -- epoch limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)
            self.session.close()

    @staticmethod
    def log(message):
        sys.stdout.write(message + "\n")
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
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
        if not os.path.exists(extracted_dir_path):
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    t = Train()
    t.train()


if __name__ == '__main__':
    tf.app.run()
