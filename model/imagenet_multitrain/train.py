import sys

import tensorflow as tf
from tensorflow.contrib import slim

from model.imagenet_multitrain.datasets import imagenet
from model.imagenet_multitrain.preprocessing import inception_preprocessing
from separable_resnet import SeparableResnet

flags = tf.app.flags
FLAGS = flags.FLAGS

image_size = 224
flags.DEFINE_integer('batch_size', 16, 'Size of each training batch')
flags.DEFINE_string('dataset_dir', '/Users/erdicalli/dev/workspace/data/imagenet', 'Size of each training batch')

# Preprocessing Flags (only affect training data, not validation data)
CHECKPOINT_FOLDER = "checkpoints"
CHECKPOINT_STEP = 800
CHECKPOINT_NAME = "SEP-RESNET-34-imagenet"
VALIDATION_STEP = 150


class Train:
    IMAGE_SIZE = 224

    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        self.graph = tf.Graph()
        self.batch_size = FLAGS.batch_size
        with self.graph.as_default():
            self.log("defining input")
            with tf.device("/cpu:0"):
                self.training_input, self.training_label = self.get_image_label_handles('train', FLAGS.dataset_dir)
                self.valid_images, self.valid_label = self.get_image_label_handles('validation', FLAGS.dataset_dir)
                self.training_label = tf.cast(self.training_label, tf.int32)
                self.valid_label = tf.cast(self.valid_label, tf.int32)

            self.log("defining models")

            with tf.variable_scope("model_32"):
                self.model_32 = SeparableResnet(training=True, batch_size=FLAGS.batch_size, layer_1_channels=32,
                                                training_inputs=self.training_input, training_label=self.training_label,
                                                valid_images=self.valid_images, valid_label=self.valid_label)
            with tf.variable_scope("model_24"):
                self.model_24 = SeparableResnet(training=True, batch_size=FLAGS.batch_size, layer_1_channels=24,
                                                training_inputs=self.training_input, training_label=self.training_label,
                                                valid_images=self.valid_images, valid_label=self.valid_label)
            with tf.variable_scope("model_16"):
                self.model_16 = SeparableResnet(training=True, batch_size=FLAGS.batch_size, layer_1_channels=16,
                                                training_inputs=self.training_input, training_label=self.training_label,
                                                valid_images=self.valid_images, valid_label=self.valid_label)
            with tf.variable_scope("model_8"):
                self.model_8 = SeparableResnet(training=True, batch_size=FLAGS.batch_size, layer_1_channels=8,
                                               training_inputs=self.training_input, training_label=self.training_label,
                                               valid_images=self.valid_images, valid_label=self.valid_label)

            with tf.variable_scope("combined"):
                self.model_32_summary_results = {"loss": self.model_32.loss,
                                                 "top_1": self.model_32.top_1_accuracy,
                                                 "top_5": self.model_32.top_5_accuracy}
                self.model_24_summary_results = {"loss": self.model_24.loss,
                                                 "top_1": self.model_24.top_1_accuracy,
                                                 "top_5": self.model_24.top_5_accuracy}
                self.model_16_summary_results = {"loss": self.model_16.loss,
                                                 "top_1": self.model_16.top_1_accuracy,
                                                 "top_5": self.model_16.top_5_accuracy}
                self.model_8_summary_results = {"loss": self.model_8.loss,
                                                "top_1": self.model_8.top_1_accuracy,
                                                "top_5": self.model_8.top_5_accuracy}

                self.combined_loss = tf.placeholder(dtype=tf.float32)
                self.combined_top_1_accuracy = tf.placeholder(dtype=tf.float32)
                self.combined_top_5_accuracy = tf.placeholder(dtype=tf.float32)

                self.combined_summary = tf.summary.merge([
                    tf.summary.scalar('loss_total', self.combined_loss),
                    tf.summary.scalar('accuracy_top1', self.combined_top_1_accuracy),
                    tf.summary.scalar('accuracy_top5', self.combined_top_5_accuracy)])

        self.log("creating session")
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)

    def train(self):
        with self.graph.as_default():
            self.log("defining summary writers")
            test_writer_32 = tf.summary.FileWriter("summaries/test_32")
            train_writer_32 = tf.summary.FileWriter("summaries/train_32")

            test_writer_24 = tf.summary.FileWriter("summaries/test_24")
            train_writer_24 = tf.summary.FileWriter("summaries/train_24")

            test_writer_16 = tf.summary.FileWriter("summaries/test_16")
            train_writer_16 = tf.summary.FileWriter("summaries/train_16")

            test_writer_8 = tf.summary.FileWriter("summaries/test_8")
            train_writer_8 = tf.summary.FileWriter("summaries/train_8")
            self.log("defined summary writers")
            saver = tf.train.Saver(max_to_keep=10)

            self.log("initializing variables")
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
            s = 0
            try:
                while not coord.should_stop():

                    summary_32, summary_24, summary_16, summary_8, _, _, _, _, step, \
                    results_32, results_24, results_16, results_8, = self.session.run(
                        [
                            self.model_32.merged_summary_op,
                            self.model_24.merged_summary_op,
                            self.model_16.merged_summary_op,
                            self.model_8.merged_summary_op,
                            self.model_32.train_step,
                            self.model_24.train_step,
                            self.model_16.train_step,
                            self.model_8.train_step,
                            self.model_32.global_step,
                            self.model_32_summary_results,
                            self.model_24_summary_results,
                            self.model_16_summary_results,
                            self.model_8_summary_results
                        ],
                        feed_dict={self.model_32.do_validate: False,
                                   self.model_24.do_validate: False,
                                   self.model_16.do_validate: False,
                                   self.model_8.do_validate: False})

                    train_writer_32.add_summary(summary_32, step)
                    train_writer_24.add_summary(summary_24, step)
                    train_writer_16.add_summary(summary_16, step)
                    train_writer_8.add_summary(summary_8, step)

                    self.write_combined_results(train_writer_32, results_32, step)
                    self.write_combined_results(train_writer_24, results_24, step)
                    self.write_combined_results(train_writer_16, results_16, step)
                    self.write_combined_results(train_writer_8, results_8, step)

                    if step % VALIDATION_STEP == 0:
                        summary_32, summary_24, summary_16, summary_8, \
                        results_32, results_24, results_16, results_8, = self.session.run([
                            self.model_32.merged_summary_op,
                            self.model_24.merged_summary_op,
                            self.model_16.merged_summary_op,
                            self.model_8.merged_summary_op,
                            self.model_32_summary_results,
                            self.model_24_summary_results,
                            self.model_16_summary_results,
                            self.model_8_summary_results
                        ],
                            feed_dict={
                                self.model_32.do_validate: True,
                                self.model_24.do_validate: True,
                                self.model_16.do_validate: True,
                                self.model_8.do_validate: True
                            })

                        test_writer_32.add_summary(summary_32, step)
                        test_writer_24.add_summary(summary_24, step)
                        test_writer_16.add_summary(summary_16, step)
                        test_writer_8.add_summary(summary_8, step)

                        self.write_combined_results(test_writer_32, results_32, step)
                        self.write_combined_results(test_writer_24, results_24, step)
                        self.write_combined_results(test_writer_16, results_16, step)
                        self.write_combined_results(test_writer_8, results_8, step)

                        test_writer_32.flush()
                        test_writer_24.flush()
                        test_writer_16.flush()
                        test_writer_8.flush()

                    if step % CHECKPOINT_STEP == 0:
                        saver.save(self.session, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=step)
                    s = step
            except tf.errors.OutOfRangeError:
                self.log('Done training -- epoch limit reached')
            finally:
                self.log('getting a last checkpoint before dying')
                if s is not 0:
                    saver.save(self.session, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=s)
                coord.request_stop()

            coord.join(threads)
            self.session.close()

    def write_combined_results(self, writer, results, step):
        combined_summary, = self.session.run([self.combined_summary],
                                             feed_dict={
                                                 self.combined_loss: results["loss"],
                                                 self.combined_top_1_accuracy: results["top_1"],
                                                 self.combined_top_5_accuracy: results["top_5"]
                                             })
        writer.add_summary(combined_summary, step)

    @staticmethod
    def log(message):
        sys.stdout.write(message + "\n")
        sys.stdout.flush()
        pass

    def get_image_label_handles(self, split, dataset_dir):

        dataset = imagenet.get_split(split, dataset_dir)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=8,
            common_queue_capacity=6 * self.batch_size,
            common_queue_min=6 * self.batch_size - 1)
        [image, label] = provider.get(['image', 'label'])
        image = inception_preprocessing.preprocess_image(image,
                                                         height=self.IMAGE_SIZE,
                                                         width=self.IMAGE_SIZE,
                                                         is_training=(split is 'train'))
        return tf.train.shuffle_batch(
            [image, label],
            batch_size=self.batch_size,
            num_threads=32,
            capacity=6 * self.batch_size,
            min_after_dequeue=6 * self.batch_size - 1)


def main(_):
    t = Train()
    t.train()


if __name__ == '__main__':
    tf.app.run()
