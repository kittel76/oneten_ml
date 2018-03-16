from __future__ import absolute_import, division, print_function

import os.path
import re
import sys
import tarfile
import glob
import json
import psutil
from collections import defaultdict
import numpy as np
import urllib
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'model_dir', '/data/www/oneten/data_find_similar_image/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

tf.app.flags.DEFINE_string(
    'output_graph_dir', '/data/www/oneten/data_find_similar_image/train_dir',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            # FLAGS.output_graph_dir, 'output_graph.pb'), 'rb') as f:
        FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_images(image_list, output_dir):
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
        # softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

        for image_index, image in enumerate(image_list):
            try:
                print("parsing", image_index, image, "\n")
                if not tf.gfile.Exists(image):
                    tf.logging.fatal('File does not exist %s', image)

                with tf.gfile.FastGFile(image, 'rb') as f:
                    image_data = f.read()

                    predictions = sess.run(softmax_tensor,
                                           {'DecodeJpeg/contents:0': image_data})

                    predictions = np.squeeze(predictions)

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

                    print("predictions", predictions)



                # close the open file handlers
                proc = psutil.Process()
                open_files = proc.open_files()

                for open_file in open_files:
                    file_handler = getattr(open_file, "fd")
                    os.close(file_handler)
            except:
                print('could not process image index', image_index, 'image', image)

    return image_to_labels

def main(_):
    output_dir = "/data/www/oneten/data_find_similar_image/image_vectors"
    images = glob.glob("/data/www/oneten/data_find_similar_image/data_dir/*/*.*")
    print("images", images)
    image_to_labels = run_inference_on_images(images, "image_vectors")
    print("image_to_labels", image_to_labels)

    # with open("image_to_labels.json", "w") as img_to_labels_out:
    #     json.dump(image_to_labels, img_to_labels_out)

    print("all done")



if __name__ == '__main__':
    tf.app.run()