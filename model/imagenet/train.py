import sys

import tensorflow as tf
import time

from separable_resnet import SeparableResnet

flags = tf.app.flags
FLAGS = flags.FLAGS

image_size = 224
flags.DEFINE_integer('batch_size', 256, 'Size of each training batch')
flags.DEFINE_string('dataset_dir', '', 'Size of each training batch')

# Preprocessing Flags (only affect training data, not validation data)
CHECKPOINT_FOLDER = "checkpoints"
CHECKPOINT_STEP = 5000
CHECKPOINT_NAME = "SEP-RESNET-34-imagenet"
VALIDATION_STEP = 150



class Train:
    def __init__(self):
        gpu_options = tf.GPUOptions()

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.model = SeparableResnet(training=True, batch_size=FLAGS.batch_size, dataset_dir=FLAGS.dataset_dir)

        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)

    def train(self):
        with self.graph.as_default():
            self.session.run(tf.variables_initializer(tf.local_variables()))
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("summaries/train", self.graph, flush_secs=30)
            test_writer = tf.summary.FileWriter("summaries/test", self.graph, max_queue=1)
            saver = tf.train.Saver(max_to_keep=10)

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
            timer = 0
            try:
                while not coord.should_stop():
                    timer += self.get_millis()
                    m, _, loss, step, = self.session.run([merged,
                                                          self.model.train_step,
                                                          self.model.loss,
                                                          self.model.global_step],
                                                         feed_dict={self.model.do_validate: False})

                    train_writer.add_summary(m, step)
                    timer -= self.get_millis()

                    if step % VALIDATION_STEP == 0:
                        m, top1, = self.session.run([merged, self.model.top_1_accuracy],
                                                    feed_dict={self.model.do_validate: True})
                        test_writer.add_summary(m, step)
                        self.log(str(timer / VALIDATION_STEP))
                        timer = 0
                    if step % CHECKPOINT_STEP == 0:
                        saver.save(self.session, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=step)
                    s = step
            except tf.errors.OutOfRangeError:
                self.log('Done training -- epoch limit reached')
            finally:
                if s != 0:
                    self.log('getting a last checkpoint before dying')
                    saver.save(self.session, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=s)
                coord.request_stop()

            coord.join(threads)
            self.session.close()

    @staticmethod
    def log(message):
        sys.stdout.write(message + "\n")
        pass

    @staticmethod
    def get_millis():
        return int(round(time.time() * 1000))

def main(_):
    t = Train()
    t.train()


if __name__ == '__main__':
    tf.app.run()
