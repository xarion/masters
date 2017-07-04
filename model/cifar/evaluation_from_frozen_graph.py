from __future__ import print_function

import cPickle
import sys
import time

import numpy as np
import tensorflow as tf

import data as cifar10
from graph_helper import read_graph

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'Size of each training batch')

DATA_DIR = "./data"
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

max_sample_count = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

IMAGE_SIZE = 32


class Evaluation:
    def __init__(self, frozen_graph):
        self.frozen_graph = frozen_graph
        self.batch_size = FLAGS.batch_size
        self.graph = read_graph(frozen_graph)
        self.session = tf.Session(graph=self.graph)
        self.input_placeholder = self.graph.get_tensor_by_name("Placeholder_1:0")
        self.logits = self.graph.get_tensor_by_name("output/BiasAdd:0")

    def evaluate(self):
        accuracies = []
        sample_count = 0
        a = self.get_validation_dataset()

        DURATION = 0
        while sample_count < max_sample_count:
            images = self.preprocess(a['data'][sample_count:(sample_count + self.batch_size)])
            labels = a['labels'][sample_count:(sample_count + self.batch_size)]
            DURATION += time.time()
            logit_values, = self.session.run([self.logits], feed_dict={self.input_placeholder: images})
            DURATION -= time.time()
            accuracy = np.mean(np.argmax(logit_values, axis=1) == labels)

            accuracies.append(accuracy)
            sample_count += FLAGS.batch_size
            # self.log("processed: %d/%d, accuracy: %.4f" % (sample_count, max_sample_count, accuracy))

        self.log("top-1 accuracy: %.4f" % (np.mean(np.array(accuracies))))
        self.log("%s \t duration: %.4f" % (self.frozen_graph, DURATION))

    @staticmethod
    def preprocess(image_batch):
        image_batch = np.reshape(image_batch, [-1, 3, 32, 32])
        image_batch = image_batch[:, (32 - IMAGE_SIZE) / 2:(32 + IMAGE_SIZE) / 2,
                      (32 - IMAGE_SIZE) / 2:(32 + IMAGE_SIZE) / 2, :]
        image_batch = np.reshape(image_batch, [-1, IMAGE_SIZE * IMAGE_SIZE])
        mean = np.mean(image_batch, axis=1)
        stddev = np.std(image_batch, axis=1)
        adjusted_stddev = np.maximum(stddev, np.ones(list(stddev.shape)) * (1. / IMAGE_SIZE))
        images = ((image_batch.transpose() - mean.transpose()) / adjusted_stddev).transpose()
        images = np.reshape(images, [-1, 3, IMAGE_SIZE, IMAGE_SIZE])
        images = np.moveaxis(images, 1, 3)
        return images

    @staticmethod
    def log(message):
        sys.stdout.write(message)
        sys.stdout.write("\n")
        sys.stdout.flush()
        pass

    @staticmethod
    def get_validation_dataset():
        with open('data/cifar-10-batches-py/test_batch', 'rb') as fo:
            dict = cPickle.load(fo)
        return dict


def main(_):
    frozen_graphs = [
        "separable_resnet-cifar-10-new-threshold-0.005.pb",
        "separable_resnet-cifar-10-new-threshold-0.002.pb",
        "separable_resnet-cifar-10-new-threshold-0.001.pb",
        "separable_resnet-cifar-10-new-threshold-0.0005.pb",
        "separable_resnet-cifar-10-new-threshold-0.0001.pb"
    ]
    for frozen_graph in frozen_graphs:
        e = Evaluation(frozen_graph)
        e.evaluate()
        e.evaluate()


if __name__ == '__main__':
    tf.app.run()
