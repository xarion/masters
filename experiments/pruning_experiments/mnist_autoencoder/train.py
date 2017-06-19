import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from experiments.pruning_experiments.mnist_autoencoder.autoencoder import Autoencoder

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


class Train:
    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.model = Autoencoder(training=True, batch_size=BATCH_SIZE)

        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True),
                                  graph=self.graph)

    def train(self):
        with self.graph.as_default():
            self.session.run(tf.variables_initializer(tf.local_variables()))
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("summaries/train", self.graph)
            test_writer = tf.summary.FileWriter("summaries/test", self.graph)
            saver = tf.train.Saver(max_to_keep=5)

            latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
            self.session.run(tf.variables_initializer(tf.local_variables()))

            if latest_checkpoint:
                self.log("loading from checkpoint file: " + latest_checkpoint)
                saver.restore(self.session, latest_checkpoint)
            else:
                self.log("checkpoint not found, initializing variables.")
                self.session.run(tf.variables_initializer(tf.global_variables()))

            self.session.run(tf.assign(self.model.global_step, 0))
            print self.session.run(self.model.learning_rate)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
            try:
                while mnist.train.epochs_completed < EPOCHS:
                    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                    m, _, loss, step, = self.session.run([merged,
                                                          self.model.train_step,
                                                          self.model.loss,
                                                          self.model.global_step],
                                                         feed_dict={self.model.do_validate: False,
                                                                    self.model.input: batch_xs})

                    train_writer.add_summary(m, step)
                    validation_batch_xs, validation_batch_ys = mnist.validation.next_batch(BATCH_SIZE)
                    m, l, = self.session.run([merged, self.model.loss],
                                             feed_dict={self.model.do_validate: True,
                                                        self.model.input: validation_batch_xs})
                    test_writer.add_summary(m, step)

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
    t = Train()
    t.train()


if __name__ == '__main__':
    tf.app.run()


    # encode_decode = sess.run(
    #     y_pred, feed_dict={X: mnist.test.images[:BATCH_SIZE]})
    # if FLAGS.plot_figures:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     f, a = plt.subplots(2, 10, figsize=(10, 2))
    #     for i in range(EXAMPLES_TO_SHOW):
    #         a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #         a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    #     f.show()
    #     plt.draw()


    # if FLAGS.plot_figures:
    #     plt.figure()
    #     num_cycles = len(node_count_histories[0])
    #     layer_1, = plt.plot(range(0, num_cycles), node_count_histories[0], label='Layer 1')
    #     layer_2, = plt.plot(range(0, num_cycles), node_count_histories[1], label='Layer 2')
    #     layer_3, = plt.plot(range(0, num_cycles), node_count_histories[2], label='Layer 3')
    #     plt.legend(handles=[layer_1, layer_2, layer_3])
    #     plt.title('Number of features per layer over time')
    #     plt.xlabel('Training Cycle')
    #     plt.ylabel('Remaining Features')
    #
    #     plt.figure()
    #     plt.plot(range(1, num_cycles), validation_loss_history)
    #     plt.title('Validation Loss over Cycles')
    #     plt.xlabel('Training Cycle')
    #     plt.ylabel('Loss')
    #
    #     plt.figure()
    #     plt.plot(range(1, num_cycles), step_durations)
    #     plt.title('Duration of Training Cycles over time')
    #     plt.xlabel('Training Cycle')
    #     plt.ylabel('Seconds')
    #     plt.show()
