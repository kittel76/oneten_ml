# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask
from flask import request

import random, json, glob, os, codecs, random
from annoy import AnnoyIndex
from scipy import spatial
import numpy as np
import time
import urllib
import tensorflow as tf
from find_similar_image import dbJob
import pandas as pd

app = Flask(__name__)


start = time.time()

# data structures
file_index_to_file_name = {}
file_index_to_file_vector = {}
chart_image_positions = {}


# config
dims = 2048
n_nearest_neighbors = 4
#trees = 10000
trees = 1000
infiles = glob.glob('/data/www/oneten/data_find_similar_image/image_vectors/*.npz')


print("trees", trees)
end = time.time() - start
print("step2", end)


# build ann index
t = AnnoyIndex(dims)
data_size = infiles.__len__()
for file_index, i in enumerate(infiles):

    #  if file_index > 10000:
    #   break;
    if file_index % 1000 == 0:
        print(file_index, i)
    file_vector = np.loadtxt(i)
    if (file_vector.__len__() == 2048):
        file_name = os.path.basename(i).split('.')[0]
        file_index_to_file_name[file_index] = file_name
        t.add_item(file_index, file_vector)
    else:
        print("not 2048")

end = time.time() - start
print("step3: before build", end)

t.build(trees)

end = time.time() - start
print("step4: after build", end)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
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


create_graph()

@app.route('/api/getSimilarImages')
def getSimilarImages ():

    img_url = request.args.get('img_url')
    if img_url == None:
        img_url = "lll"

    rand = random.randint(1, 100000)
    tmpFileName = "/tmp/img_" + rand.__str__() + ".jpg"
    tmpResizeFileName = "/tmp/img_resize_" + rand.__str__() + ".jpg"

    tmpFileName, _ = urllib.urlretrieve(img_url, tmpFileName)

    with tf.Session() as sess:
        with tf.gfile.FastGFile(tmpFileName, 'rb') as f:
            image_data = f.read()

        feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        feature_set = sess.run(feature_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        feature_vector = np.squeeze(feature_set)
        outfile_name = os.path.basename(tmpFileName) + ".npz"
        out_path = os.path.join("/tmp/", outfile_name)
        np.savetxt(out_path, feature_vector, delimiter=',')


    print("out_path", out_path)
    test_file_vector = np.loadtxt(out_path)

    idx = file_index
    file_index_to_file_name[idx] = os.path.basename(out_path).split('.')[0]

    file_index_to_file_vector[idx] = test_file_vector
    t.add_item(idx, test_file_vector)


    t.build(trees)

    print("build end !!")

    named_nearest_neighbors = []
    nearest_neighbors = t.get_nns_by_item(idx, n_nearest_neighbors)
    print("nearest_neighbors", nearest_neighbors)


    image_urls = []
    sim_arrs = []
    image_strs = []

    # image_urls[0] = img_url


    index = 0
    for j in nearest_neighbors:

        neighbor_file_vector = t.get_item_vector(j)
        similarity = 1 - spatial.distance.cosine(t.get_item_vector(idx), neighbor_file_vector)
        rounded_similarity = int((similarity * 10000)) / 10000.0

        print("file_index_to_file_name[j]", file_index_to_file_name[j])

        print("index", index)
        if j == idx:
            # image_urls.append(img_url)
            print("original")
        else:
            print("kk", file_index_to_file_name[j])
            img = dbJob.getImg(file_index_to_file_name[j])
            sim_arrs.append(rounded_similarity)
            image_urls.append( img)
            image_strs.append( "<img src='" + image_urls[index] + "' width=300 height=300 />")
            index = index+1


    print("image_urls", image_urls)
    print("sim_arrs", sim_arrs)

    str_list = []
    str_list.append("<table><tr><td colspan=4><img src='" + img_url + "' width=300 height=300 /></td></tr>")
    str_list.append("<tr>")

    for i in xrange(image_urls.__len__()):
        str_list.append("<td>" + image_strs[i] + "</td>")
    str_list.append("</tr>")
    str_list.append("<tr>")
    for i in xrange(image_urls.__len__()):
        str_list.append("<td>"+str(sim_arrs[i]) + "</td>")
    str_list.append("</tr>")
    str_list.append("</table>")
    return ''.join(str_list)



if __name__ == '__main__':

    app.run("0.0.0.0", "8001")
