# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os



from similar_img_prd import dbJob
import tensorflow as tf
import urllib
import time
import shutil
import logging.config
import logging


logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('similar_img_prd')

tf.app.flags.DEFINE_string('data_directory', '/data/www/oneten/data_similar_img_prd/data_dir/',
                           'Output data directory')
tf.app.flags.DEFINE_string('data_rep_directory', '/data/www/oneten/data_similar_img_prd/data_dir_rep/',
                           'Output data directory')




FLAGS = tf.app.flags.FLAGS

global already_exists_data_cnt
global created_data_cnt


def _make_data(directory, rep):

    ## 디렉토리 없으면 생성 있으면 삭제하고 재생성
    if  os.path.isdir(directory):
        shutil.rmtree(directory)
        os.mkdir(directory)


    if not os.path.isdir(rep):
        os.mkdir(rep)

    cate_nms, cate_nos = dbJob.getCateNms()

    already_exists_data_cnt = 0
    created_data_cnt = 0
    ## repository
    for (i, cate_no) in zip(cate_nms, cate_nos):
        try:
            os.stat(rep+i)
        except:
            os.mkdir(rep+ i)

    for (i, cate_no) in zip(cate_nms, cate_nos):
        try:
            os.stat(directory+i)
        except:
            os.mkdir(directory+ i)

        img_list, file_list = dbJob.getCatePrdImgList(cate_no)
        for url, filename in zip(img_list, file_list):

            if os.path.exists(rep+ i + "/"  + filename):
                already_exists_data_cnt = already_exists_data_cnt+1
                shutil.copyfile(rep+ i + "/"  + filename, directory+ i + "/"  + filename )
            else:
                if not os.path.exists(directory+ i + "/"  + filename):
                    try:

                        filename_, _ = urllib.urlretrieve(url, directory + i + "/" + filename)
                        shutil.copyfile(filename_, rep + i + "/" + filename)

                        created_data_cnt = created_data_cnt +1
                    except Exception as e:
                        print(e)
                        print('SKIPPED: urllib.urlretrieve eror while dowwnload %s.' % filename)
                        # os.remove(filename)
                        continue

    logger.info("already exists cnt: %s", already_exists_data_cnt)
    logger.info("created_data_cnt: %s", created_data_cnt)

def main(unused_argv):


  logger.info("start")
  print('Saving results to %s' % FLAGS.data_directory)
  start = time.time()

  _make_data(FLAGS.data_directory, FLAGS.data_rep_directory)

  end = time.time() - start
  logger.info("end")
  logger.info("elapsed time: %s", end)

if __name__ == '__main__':
  tf.app.run()

