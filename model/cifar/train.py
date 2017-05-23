import os
import sys
import tarfile

import tensorflow as tf
from six.moves import urllib

import data as cifar10
from model import SeparableResnet

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('gpu', 1, '0:=GTX 1080 TI, 1:=GTX 980 TI')
flags.DEFINE_integer('mode', 1, '1:=train, 2:=eval, 3:=save')
flags.DEFINE_integer('max_steps', 10000000, 'Number of steps to run trainer.')
flags.DEFINE_integer('column', 1, 'Column number for .csv labels file.')
flags.DEFINE_integer('test_size', 50000, 'Size of test set.')
flags.DEFINE_integer('batch_size', 32, 'Size of each training batch')
flags.DEFINE_integer('image_size', 350, 'Size of input images')
flags.DEFINE_integer('input_size', 224, 'Size of input images')
flags.DEFINE_integer('input_type', 1, 'Type of input (1:image, 2:vector/bottleneck)')
flags.DEFINE_integer('val_freq', 1000, 'Validation every x iterations')
flags.DEFINE_integer('test_batch_size', 128, 'Size of each test batch')
flags.DEFINE_integer('balanced', 0, 'Size of each test batch')
flags.DEFINE_integer('seed', 123456, 'Random seed')
flags.DEFINE_float('learning_rate', None, 'Initial learning rate.')
flags.DEFINE_float('weight_decay', None, 'General weight decay')
flags.DEFINE_float('dropout', 0.8, 'Keep probability for training dropout.')
flags.DEFINE_float('save_thres', 3.0, 'Upper error threshold - above the model is never saved.')
flags.DEFINE_string('name', 'separable_resnet', 'Name of the model')
flags.DEFINE_string('data_dir', 'data/inception/', 'Directory containing the images')
flags.DEFINE_string('labels', 'inception.csv', 'File to load labels from.')
flags.DEFINE_string('load_name', 'densenet2', 'File to load labels from.')
flags.DEFINE_boolean('continued', False, 'Whether to continue from a previous saved checkpoint.')
flags.DEFINE_boolean('fullpath', False, 'Whether labels are given as full path or just filenames')
flags.DEFINE_boolean('hist', False, 'Whether to track the histograms in tensorboard - without it is faster')
flags.DEFINE_boolean('log', False, 'Whether to log an extra .csv file - for multi training')
flags.DEFINE_boolean('summary', True, 'Whether to write summary files - not necessary in multi training')

# Preprocessing Flags (only affect training data, not validation data)
flags.DEFINE_boolean('flip', True, 'Random flips of training images')
flags.DEFINE_boolean('crop', True,
                     'Randomly resizes the image between image_size and input_size, then crops to input_size')
flags.DEFINE_boolean('color', False, 'Randomly affects color aspects - brightness, contrast, hue, saturation')
flags.DEFINE_boolean('rotate', False, 'Randomly rotates the image')

CHECKPOINT_FOLDER = "checkpoints"
CHECKPOINT_STEP = 50000
CHECKPOINT_NAME = "SEP-RESNET-52"

DATA_DIR = "./data"
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


class Train:
    def __init__(self):
        available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES'] = available_devices[FLAGS.gpu]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        self.graph = tf.Graph()

        with self.graph.as_default():
            if FLAGS.mode == 1:
                self.maybe_download_and_extract()
                train_images, train_labels = cifar10.distorted_inputs(DATA_DIR + "/cifar-10-batches-bin",
                                                                      FLAGS.batch_size)
                self.model = SeparableResnet(learning_rate=FLAGS.learning_rate,
                                             input_tensor=train_images,
                                             label_tensor=train_labels)

            elif FLAGS.mode == 2:
                valid_images, valid_labels = cifar10.inputs(True, DATA_DIR + "/cifar-10-batches-bin", FLAGS.batch_size)

                self.model = SeparableResnet(learning_rate=FLAGS.learning_rate,
                                             input_tensor=valid_images,
                                             label_tensor=valid_labels)
            else:
                self.model = SeparableResnet()

        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True),
                                  graph=self.graph)

    def train(self):
        with self.graph.as_default():
            self.session.run(tf.variables_initializer(tf.local_variables()))
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("summaries", self.graph)
            saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=4)

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
                    m, _, loss, step, = self.session.run(
                        [merged,
                         self.model.train_step,
                         self.model.loss,
                         self.model.global_step])

                    train_writer.add_summary(m, step)
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
        # logger.info(message)
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
