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

"""Evaluation for CIFAR-10.
Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.
Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from img_style_process2 import cifar10

FLAGS = tf.app.flags.FLAGS


CHECKPOINT_DIR = '/data/www/oneten/dl_img_style_process2/train_dir_for_service/'


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image



def get_process_image(filename):

  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  coder = ImageCoder()
  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  reshaped_image = tf.cast(image, tf.float32)
  height = 48
  width = 48

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])


  images = tf.train.batch(
    [float_image],
    batch_size=128)

  return images



def getStyle(filename):
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images = get_process_image(filename)

    # images.set_shape([128])
    # labels.set_shape([128])

    # Build a Graph that computes the logits predictions from the
    # inference model.


    logits = cifar10.inference(images)
    hypothesis = tf.nn.softmax(logits)
    prediction = tf.argmax(hypothesis, 1)

    print("hypothesis", hypothesis)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
      cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    return eval(saver, images, prediction, hypothesis)



def eval(saver, images, prediction, hypothesis):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    print("CHECKPOINT_DIR", CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))


      predictions_, images_, hypothesis_ = sess.run([prediction, images, hypothesis])


      ranking = np.argsort(hypothesis_)
      print("ranking", ranking[0])

      # b[b.__len__() - 1]


    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return ranking[0][::-1]

def main(argv=None):  # pylint: disable=unused-argument

  # evaluate()
  # showImg()
  # oneImg()

  print("aa")


  first, second = getStyle("/tmp/img_resize_84782.jpg")
  print("in main:",  first, second)


  unique_labels = [l.strip() for l in tf.gfile.FastGFile('/data/www/oneten/dl_img_style_process2/data_dir/labels.txt', 'r').readlines()]
  dictionary = dict()
  for idx in range(unique_labels.__len__()):
    dictionary[idx] = unique_labels[idx]

  print(dictionary[first])


if __name__ == '__main__':
  tf.app.run()