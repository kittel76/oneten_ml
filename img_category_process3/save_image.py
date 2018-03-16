# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import threading

import numpy as np
import tensorflow as tf
from img_category_process3 import dbJob
import sys

import urllib
import shutil
from PIL import Image
import time
from db_util import log_util

tf.app.flags.DEFINE_string('train_directory', '/data/www/oneten/dl_img_category_process3/data_dir/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/data/www/oneten/dl_img_category_process3/data_dir/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/data/www/oneten/dl_img_category_process3/data_dir',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 3,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 3,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 3,
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string('labels_file', '/data/www/oneten/dl_img_category_process3/data_dir/labels.txt', 'Labels file')


FLAGS = tf.app.flags.FLAGS

def _is_png(filename):
  """Determine if a file contains a PNG format image.
  Args:
    filename: string, path of the image file.
  Returns:
    boolean indicating if the image is a PNG.
  """
  # return '.png' in filename
  return 0

def _process_image(filename, coder, ilabel):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()


  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)


  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # resize_image = tf.image.resize_image_with_crop_or_pad(image=image,
  #                                                        target_height=600,
  #                                                        target_width=600)
  # sess =  tf.Session()
  # image = sess.run(resize_image)


  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3



  r = image[:, :, 0].flatten()
  g = image[:, :, 1].flatten()
  b = image[:, :, 2].flatten()
  label = [ilabel]

  out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
  # print("filename", a)
  # return resized_image, 100, 100
  # return image.tostring(), height, width
  return image_data, height, width, out




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




def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.
  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0



  for s in range(num_shards_per_batch):
    print("num_shards_per_batch", num_shards_per_batch)
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)

    w = open(output_file, "wb")
    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    print("files_in_shard", files_in_shard)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      text = texts[i]


      try:
        image_buffer, height, width, out = _process_image(filename, coder, label)
      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected eror while decoding %s.' % filename)
        continue

      if  (width, height) == (128,128):
          w.write(out)
          shard_counter += 1
          counter += 1



      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    w.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):
  """Process and save list of images as TFRecord of Example protos.
  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []

  print("len(filenames)", len(filenames))
  print("spacing", spacing)
  print("spacing:", len(spacing) - 1)
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  print("ranges", ranges)

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    print("thread_index", ranges)
    args = (coder, thread_index, ranges, name, filenames,
            texts, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
  """Build a list of all images files and labels in the data set.
  Args:
    data_dir: string, path to the root directory of images.
      Assumes that the image data set resides in JPEG files located in
      the following directory structure.
        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg
      where 'dog' is the label associated with these images.
    labels_file: string, path to the labels file.
      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        dog
        cat
        flower
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.
  Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)

  unique_labels = [l.strip() for l in tf.gfile.FastGFile(
      labels_file, 'r').readlines()]

  print("labels_file", labels_file)
  print("unique_labels", unique_labels)

  labels = []
  filenames = []
  texts = []

  # Leave label index 0 empty as a background class.
  label_index = 0

  # Construct the list of JPEG files and labels.
  for text in unique_labels:
    jpeg_file_path = '%s/%s/*' % (data_dir, text)
    # print("jpeg_file_path", jpeg_file_path)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    # print("matching_files", len(matching_files), label_index)

    labels.extend([label_index] * len(matching_files))
    texts.extend([text] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1


  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  texts = [texts[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]



  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file):
  """Process a complete data set and save it as a TFRecord.
  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  filenames, texts, labels = _find_image_files(directory, labels_file)



  # print("file info", filenames, texts, labels)
  print("name, directory, num_shards, labels_file",name, directory, num_shards, labels_file )
  _process_image_files(name, filenames, texts, labels, num_shards)
  log_util.update_job_log(job_code="CATEGORY3", data_cnt=len(filenames))

def _make_data(directory, labels_file):
    # os.rmdir(directory)
    # os.mkdir(directory)


    # shutil.rmtree(directory)
    if not os.path.isdir(directory):
        os.mkdir(directory)



    cate_nms, cate_nos = dbJob.getCateNms()
    print(cate_nms)
    with open(labels_file, "w") as f:
        for (i, cate_no) in zip(cate_nms, cate_nos):
            f.write(i + os.linesep)
    for (i, cate_no) in zip(cate_nms, cate_nos):
        try:
            os.stat(directory+i)
        except:
            os.mkdir(directory+ i)

        img_list, file_list = dbJob.getCatePrdImgList(cate_no)
        for url, filename in zip(img_list, file_list):
            # print("filename", filename)
            if not os.path.exists(directory+ i + "/"  + filename):
                try:
                    filename, _ = urllib.urlretrieve(url, directory + i + "/" + filename)
                except Exception as e:
                    print(e)
                    print('SKIPPED: urllib.urlretrieve eror while dowwnload %s.' % filename)
                    # os.remove(filename)
                    continue


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)
  start = time.time()

  _make_data(FLAGS.validation_directory, FLAGS.labels_file)
  _process_dataset('train-data', FLAGS.train_directory,
                   FLAGS.train_shards, FLAGS.labels_file)


  end = time.time() - start
  print("time", end)


if __name__ == '__main__':
  tf.app.run()

