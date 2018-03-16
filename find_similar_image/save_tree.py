import cPickle as pickle
from annoy import AnnoyIndex
import random, json, glob, os, codecs, random
import numpy as np
import pandas as pd
import tensorflow as tf


# data structures
file_index_to_file_name = {}
file_index_to_file_vector = {}
chart_image_positions = {}

file_arr = []


# config
dims = 2048
n_nearest_neighbors = 10
trees = 10000
infiles = glob.glob('/data/www/oneten/data_find_similar_image/image_vectors/*.npz')


t = AnnoyIndex(dims)
data_size = 0
for file_index, i in enumerate(infiles):
  if file_index >100000:
      break;
  if file_index % 100 == 100:
    print(file_index, i)
  file_vector = np.loadtxt(i)
  if(file_vector.__len__() == 2048):
      file_name = os.path.basename(i).split('.')[0]
      file_index_to_file_name[file_index] = file_name
      file_index_to_file_vector[file_index] = file_vector
      t.add_item(file_index, file_vector)
      file_arr.append(file_name)
      data_size = data_size + 1
  else:
    print("not 2048")

print("data_size", data_size)
# t.build(trees)

# t.save('/data/www/oneten/data_find_similar_image/tree_data/data.tree')


print("file_arr", file_arr)

my_pd = pd.DataFrame(file_arr)
my_pd.to_csv('/data/www/oneten/data_find_similar_image/tree_data/data.csv', header=['file_name'])




def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            # FLAGS.output_graph_dir, 'output_graph.pb'), 'rb') as f:
            "/tmp/imagenet", 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


create_graph()


# u = AnnoyIndex(dims)
# # u.build(trees)
# u.load('/data/www/oneten/data_find_similar_image/tree_data/data.tree')
#
#

tmpFileName = "./860953.jpg"
with tf.Session() as sess:
    with tf.gfile.FastGFile(tmpFileName, 'rb') as f:
        image_data = f.read()

    feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    feature_set = sess.run(feature_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    feature_vector = np.squeeze(feature_set)
    outfile_name = os.path.basename(tmpFileName) + ".npz"
    out_path = os.path.join("./", outfile_name)
    np.savetxt(out_path, feature_vector, delimiter=',')

    print("out_path", out_path)

    test_file_vector = np.loadtxt(out_path)

    t.unbuild()
    t.add_item(894, test_file_vector)



#
#
# nearest_neighbors = u.get_nns_by_item(0, n_nearest_neighbors)
# print("nearest_neighbors", nearest_neighbors)
#
# print(u.get_item_vector(0))
# print(u.get_item_vector(1))
# print(u.get_item_vector(894))