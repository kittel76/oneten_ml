# coding=utf-8
#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of Estimator for DNN-based text classification with DBpedia data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
from text_style_classify import common

FLAGS = None

MAX_DOCUMENT_LENGTH = common.MAX_DOCUMENT_LENGTH
EMBEDDING_SIZE = common.EMBEDDING_SIZE
MAX_LABEL = common.MAX_LABEL
WORDS_FEATURE = common.WORDS_FEATURE  # Name of the input words feature.
MAX_TRAIN_STEP = common.MAX_TRAIN_STEP


def estimator_spec_for_softmax_classification(
    logits, labels, mode):
  """Returns EstimatorSpec instance for softmax classification."""
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  onehot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def bag_of_words_model(features, labels, mode):
  """A bag-of-words model. Note it disregards the word order in the text."""
  bow_column = tf.feature_column.categorical_column_with_identity(
      WORDS_FEATURE, num_buckets=n_words)
  bow_embedding_column = tf.feature_column.embedding_column(
      bow_column, dimension=EMBEDDING_SIZE)
  bow = tf.feature_column.input_layer(
      features,
      feature_columns=[bow_embedding_column])
  logits = tf.layers.dense(bow, MAX_LABEL, activation=None)

  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)


def rnn_model(features, labels, mode):
  """RNN model to predict from sequence of words to a class."""
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  word_vectors = tf.contrib.layers.embed_sequence(
      features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  # Split into list of embedding per word, while removing doc length dim.
  # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
  word_list = tf.unstack(word_vectors, axis=1)

  # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
  cell = tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)

  # Create an unrolled Recurrent Neural Networks to length of
  # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
  _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

  # Given encoding of RNN, take encoding of last step (e.g hidden size of the
  # neural network of last step) and pass it as features for softmax
  # classification over output classes.
  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)

import pandas as pd
from datetime import datetime


def getPredict(prd_nm, vocab_processor):
    global n_words

    start = datetime.now()

    print("prd_nm", prd_nm)
    x_test = pd.Series([prd_nm])
    # vocab_processor build
    # x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_test = np.array(list(x_transform_test))

    n_words = len(vocab_processor.vocabulary_)

    # Build model
    # Switch between rnn_model and bag_of_words_model to test different models.
    model_fn = rnn_model
    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=common.TRAIN_DIR_FOR_SERVICE)

    # Predict.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_test},
        shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    y_predicted = np.array(list(p['class'] for p in predictions))


    return y_predicted


def main(unused_argv):
    global n_words

    prd_nm = '러블리 스커트'

    x_test = [prd_nm]

    print("x_test", x_test)

    df = pd.read_csv(common.DATA_DIR_FOR_SERVICE + "data.csv", header=1, sep='!@!', names=["prd_nm", "idx"])
    x_train = pd.Series(df["prd_nm"])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(common.MAX_DOCUMENT_LENGTH)
    x_transform_train = vocab_processor.fit_transform(x_train)


    y_predicted = getPredict(prd_nm=prd_nm, vocab_processor=vocab_processor)
    result_batch = y_predicted[0] - 1

    print("result_batch", result_batch)

    unique_labels = [l.strip() for l in
                     tf.gfile.FastGFile(common.DATA_DIR_FOR_SERVICE + 'labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels.__len__()):
        dictionary[idx] = unique_labels[idx]

    result = "{\"cate_no\":" + str(dictionary[result_batch]).split("_")[0] + ", \"cate_nm\":\"" + \
             str(dictionary[result_batch]).split("_")[1] + "\", \"prd_nm\":\"" + prd_nm + "\"" + "}"

    print("result", result)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--test_with_fake_data',
      default=False,
      help='Test the example code with fake data.',
      action='store_true')
  parser.add_argument(
      '--bow_model',
      default=False,
      help='Run with BOW model instead of RNN.',
      action='store_true')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)