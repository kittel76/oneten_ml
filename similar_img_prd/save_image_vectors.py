# coding=utf-8
from __future__ import absolute_import, division, print_function

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


import os.path
import sys
import tarfile
import glob
from collections import defaultdict
import numpy as np
import urllib, shutil
import tensorflow as tf
import time
import logging.config
import logging

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('similar_img_prd')

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

tf.app.flags.DEFINE_string('image_vectors_dir', '/data/www/oneten/data_similar_img_prd/image_vectors/',
                           'Output data directory')

tf.app.flags.DEFINE_string('image_vectors_rep_dir', '/data/www/oneten/data_similar_img_prd/image_vectors_rep/',
                           'Output data directory')

tf.app.flags.DEFINE_string('image_data_directory', '/data/www/oneten/data_similar_img_prd/data_dir/',
                           'Output data directory')


global already_exists_data_cnt
global created_data_cnt


tf.app.flags.DEFINE_string(
    'output_graph_dir', '/data/www/oneten/data_similar_img_prd/train_dir',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
        FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_images(image_list, output_dir, rep):
    """Runs inference on an image list.

    Args:
      image_list: a list of images.
      output_dir: the directory in which image vectors will be saved

    Returns:
      image_to_labels: a dictionary with image file keys and predicted
        text label values
    """
    image_to_labels = defaultdict(list)

    create_graph()
    already_exists_data_cnt = 0
    created_data_cnt = 0
    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        # softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

        for image_index, image in enumerate(image_list):
            try:
                if not tf.gfile.Exists(image):
                    tf.logging.fatal('File does not exist %s', image)

                outfile_name = os.path.basename(image) + ".npz"
                out_path = os.path.join(output_dir, outfile_name)

                rep_out_path = os.path.join(rep, outfile_name)
                if os.path.exists(rep_out_path): ## 기존에 생성한 파일이 있으면
                    shutil.copyfile(rep_out_path, out_path)
                    already_exists_data_cnt = already_exists_data_cnt + 1
                    continue

                with tf.gfile.FastGFile(image, 'rb') as f:
                    image_data = f.read()
                    ###
                    # Get penultimate layer weights
                    ###


                    feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
                    feature_set = sess.run(feature_tensor,
                                           {'DecodeJpeg/contents:0': image_data})
                    feature_vector = np.squeeze(feature_set)
                    outfile_name = os.path.basename(image) + ".npz"
                    out_path = os.path.join(output_dir, outfile_name)
                    np.savetxt(out_path, feature_vector, delimiter=',')
                    shutil.copyfile(out_path, rep_out_path)
                    created_data_cnt = created_data_cnt + 1

            except:
                logger.error('could not process image index:%s, image:%s', image_index, image)


    logger.info("already exists cnt: %s", already_exists_data_cnt)
    logger.info("created_data_cnt: %s", created_data_cnt)
    return image_to_labels


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)
        logger.info('Succesfully downloaded: %s, %s bytes.', filename, statinfo.st_size)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    start = time.time()

    maybe_download_and_extract()

    if  os.path.exists(FLAGS.image_vectors_dir):
        shutil.rmtree(FLAGS.image_vectors_dir)
    if not os.path.exists(FLAGS.image_vectors_dir):
        os.makedirs(FLAGS.image_vectors_dir)

    if not os.path.isdir(FLAGS.image_vectors_rep_dir):
        os.mkdir(FLAGS.image_vectors_rep_dir)

    images = glob.glob(FLAGS.image_data_directory + "*/*.*")
    image_to_labels = run_inference_on_images(images, FLAGS.image_vectors_dir, FLAGS.image_vectors_rep_dir)

    end = time.time() - start
    logger.info("save_image_vectors time: %s", end)


if __name__ == '__main__':
    tf.app.run()