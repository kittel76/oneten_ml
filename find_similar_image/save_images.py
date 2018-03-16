# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os


import numpy as np
import tensorflow as tf
from find_similar_image import dbJob

import urllib, shutil
import time

import logging.config
import logging
logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('root')

tf.app.flags.DEFINE_string('image_directory', '/data/www/oneten/data_find_similar_image/data_dir/', 'image data directory')
tf.app.flags.DEFINE_string('org_image_directory', '/data/www/oneten/data_find_similar_image/org_data_dir/', 'original image data directory')


FLAGS = tf.app.flags.FLAGS

def _make_data(directory):
    shutil.rmtree(directory)
    if not os.path.isdir(directory):
        os.mkdir(directory)


    cate_nms, cate_nos = dbJob.getCateNms()
    print(cate_nms)

    for (i, cate_no) in zip(cate_nms, cate_nos):
        try:
            os.stat(directory+i)
        except:
            os.mkdir(directory+ i)

        img_list, file_list = dbJob.getCatePrdImgList(cate_no)
        for url, filename in zip(img_list, file_list):
            if not os.path.exists(directory+ i + "/"  + filename):
                try:
                    filename, _ = urllib.urlretrieve(url, directory + i + "/" + filename)
                except Exception as e:
                    print(e)
                    print('SKIPPED: urllib.urlretrieve eror while dowwnload %s.' % filename)
                    # os.remove(filename)
                    continue

def _make_data_for_thread(directory):
    shutil.rmtree(directory)
    if not os.path.isdir(directory):
        os.mkdir(directory)


    global thread_size
    global thread_result
    global thread_processing_result
    global data_list

    # img_list, file_list, cate_list, prd_list = dbJob.getPrdPrdImgList()
    # for url, filename, cate_nm, prd_no in zip(img_list, file_list, cate_list, prd_list):
    #     print(url, filename, cate_nm, prd_no, int(prd_no) %3)

    thread_size = 5
    thread_result = np.zeros(thread_size)
    thread_processing_result = np.zeros(thread_size)

    data_list = dbJob.getPrdPrdImgList()

    import threading
    for i in range(thread_size):
        threading.Thread(target=insertData, args=(i, thread_size)).start()

    while 1:
        time.sleep(5)
        logger.info("check thread:[%s]", thread_result)
        logger.info("thread process cnt:[%s]", thread_processing_result)
        if (thread_result.all() == 1):
            logger.info(" thread is end")
            break
        current_time = time.time()


def  insertData(thread_num, thread_max):

    for idx, data in enumerate(data_list):
        if (idx % thread_max == thread_num):
            url = data[0]
            filename = data[1]
            catename = data[2]
            prdno = data[3]

            org_filepath = FLAGS.org_image_directory + catename + "/" + filename
            filepath = FLAGS.image_directory + catename + "/" + filename
            if not tf.gfile.Exists(org_filepath):
                try:
                    os.stat(FLAGS.org_image_directory+ catename)
                except:
                    makeDir(FLAGS.org_image_directory+ catename)
                try:
                    filename, _ = urllib.urlretrieve(url, org_filepath)
                    shutil.copyfile(org_filepath, filepath)
                    thread_processing_result[thread_num] = thread_processing_result[thread_num] + 1
                except Exception as e:
                    print(e)
                    print('SKIPPED: urllib.urlretrieve eror while dowwnload %s.' % filename)
                    continue
            else:
                try:
                    os.stat(FLAGS.image_directory+ catename)
                except:
                    makeDir(FLAGS.image_directory+ catename)
                shutil.copyfile(org_filepath, filepath)
                thread_processing_result[thread_num] = thread_processing_result[thread_num] + 1

    thread_result[thread_num] = 1

def makeDir(filepath):
    try:
        os.mkdir(filepath)
    except:
        print("alreday exists")

def main(unused_argv):

  print('Saving results to %s' % FLAGS.image_directory)
  start = time.time()

  _make_data_for_thread(FLAGS.image_directory)


  end = time.time() - start
  print("time", end)


if __name__ == '__main__':
  tf.app.run()

