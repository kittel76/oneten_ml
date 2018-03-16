# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from annoy import AnnoyIndex
from scipy import spatial
import glob, os
import numpy as np
import tensorflow as tf
import time
import logging.config
import logging

from similar_img_prd import dbio

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('similar_img_prd')

def main(unused_argv):

    start = time.time()

    # data structures
    file_index_to_file_name = {}
    file_index_to_file_vector = {}
    chart_image_positions = {}

    # config
    dims = 2048
    n_nearest_neighbors = 41
    trees = 1000
    infiles = glob.glob('/data/www/oneten/data_similar_img_prd/image_vectors/*.npz')

    # build ann index
    t = AnnoyIndex(dims)
    for file_index, i in enumerate(infiles):
      file_vector = np.loadtxt(i)
      file_name = os.path.basename(i).split('.')[0]
      file_index_to_file_name[file_index] = file_name
      file_index_to_file_vector[file_index] = file_vector
      t.add_item(file_index, file_vector)
    t.build(trees)

    conn = dbio.getConnection()
    dbio.truncate_related_prd_list_tmp()
    for i in file_index_to_file_name.keys():
      if i%1000 == 0:
          logger.info("processed count:%s", i)

      master_file_name = file_index_to_file_name[i]
      master_vector = file_index_to_file_vector[i]

      nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)


      image_list = []

      idx = 0
      for j in nearest_neighbors:
        neighbor_file_name = file_index_to_file_name[j]
        neighbor_file_vector = file_index_to_file_vector[j]

        if idx > 0:
            image_list.append(file_index_to_file_name[j])

        similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
        rounded_similarity = int((similarity * 10000)) / 10000.0
        idx = idx +1

      dbio.insert_related_prd_list_tmp(master_file_name, image_list, conn)

    logger.info("processed total count:%s", i)


    conn.close()
    dbio.truncate_related_prd_list()
    dbio.insert_related_prd_list()

    end = time.time() - start

    logger.info("elasped time: %s", end)


if __name__ == '__main__':
  tf.app.run()