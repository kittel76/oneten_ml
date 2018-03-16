# coding=utf-8
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
from tensorflow.python.framework import graph_util
import os
import shutil
from web import image_util

from img_detail_process2 import cifar10


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('export_path_base', '/data/www/oneten/dl_img_detail_process2/model/',
                           """Directory where to read model checkpoints.""")



def main(argv=None):  # pylint: disable=unused-argument


    input_real=tf.placeholder(tf.float32, shape=(128, 48, 48,3))
    logits = cifar10.inference(input_real)
    hypothesis = tf.nn.softmax(logits)
    prediction = tf.argmax(hypothesis, 1)


    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    sess = tf.InteractiveSession()

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return




    export_path = os.path.join(
        tf.compat.as_bytes(FLAGS.export_path_base),
        tf.compat.as_bytes("1"))
    print('Exporting trained model to', export_path)

    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)


    # Build the signature_def_map.
    classification_inputs = tf.saved_model.utils.build_tensor_info(
        input_real)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
        prediction)

    classification_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={
              "image": classification_inputs
          },
          outputs={
              "class": classification_outputs_classes
          },
          method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
    sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
      'predict_images':
          classification_signature,
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          classification_signature,
    },
    legacy_init_op=legacy_init_op)

    builder.save()



if __name__ == '__main__':
  tf.app.run()