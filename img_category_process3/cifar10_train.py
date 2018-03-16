# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
import time

import tensorflow as tf
import numpy as np

from img_category_process3 import cifar10
from db_util import log_util
from tensorflow.python.framework import graph_util

FLAGS = tf.app.flags.FLAGS




tf.app.flags.DEFINE_string('train_dir', '/data/www/oneten/dl_img_category_process3/train_dir/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
# tf.app.flags.DEFINE_integer('max_steps', 1000000,
#                             """Number of batches to run.""")
tf.app.flags.DEFINE_integer('max_steps', 60000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 50,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        print("_LoggerHook begin")
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * cifar10.BATCH_SIZE / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      print("monitored TrainsingSession")
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
  log_util.update_job_log(job_code="CATEGORY3", traing_step=FLAGS.max_steps, batch_size=cifar10.BATCH_SIZE)

def main(argv=None):  # pylint: disable=unused-argument

  train()
  # freeze_graph(FLAGS.train_dir)
  # graph = load_graph(FLAGS.train_dir + "frozen_model.pb")

  # for op in graph.get_operations():
  #     print(op.name)
  #
  # logits = graph.get_tensor_by_name('prefix/softmax_linear/softmax_linear:0')
  # hypothesis = tf.nn.softmax(logits)
  # prediction = tf.argmax(hypothesis, 1)
  # print("hypothesis", hypothesis, logits)
  #
  # filename = "/tmp/img_75918.jpg"
  #
  # images_ = get_process_image(filename)
  #
  # with tf.Session(graph=graph) as sess:
  #   coord = tf.train.Coordinator()
  #   print("kkkk")
  #   # Start the queue runners.
  #   #
  #   hypothesis_ = sess.run(hypothesis, feed_dict={images: images_})
  #   print("kkkk!!!")
  #   #   print("kkkkk")
  #   #   ranking = np.argsort(hypothesis_)
  #   #   print("ranking", ranking[0])




if __name__ == '__main__':
  tf.app.run()