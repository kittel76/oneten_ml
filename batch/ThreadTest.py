from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from collections import deque

import numpy as np
import tensorflow as tf


from batch import dataio
from batch import ops
from batch import dbio
import sys
np.random.seed(13575)

DIM = ops.DIM
DEVICE = "/cpu:0"


from datetime import date
import logging.config
import logging
logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('batch')

targetDate = dbio.getTargetDate()

logger.info ("targetDate:%s", targetDate)
if targetDate == None:
    logger.error("targetDate is None")
    sys.exit()


target_data_dir = "/data/www/oneten/dl_prd_rating/data_dir/" + targetDate
target_train_dir = "/data/www/oneten/dl_prd_rating/train_dir/" + targetDate


def get_data():
    df = dataio.read_process(target_data_dir + "/click.csv", sep=",")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test, df


def svd(total):


    global  thread_result
    thread_result = np.array( [False, False, False, False])

    dbio.truncate_user_item_rank_tmp()

    import threading

    t1 = threading.Thread(target=insertData, args=(0, 4,  total))
    t1.start()

    t2 = threading.Thread(target=insertData, args=(1, 4,  total))
    t2.start()

    t3 = threading.Thread(target=insertData, args=(2, 4,  total))
    t3.start()

    t4 = threading.Thread(target=insertData, args=(3, 4,  total))
    t4.start()

    while 1:
        time.sleep(1)
        logger.info("check thread:[%s]",  thread_result )
        if(thread_result.any() == True ):
            logger.info("0 thread is end")
            break


def insertData(index, thread_max,  total):
    insertData_start_time = time.time()

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    USER_NUM = dbio.getMaxUserNum(targetDate) + 1
    ITEM_NUM = dbio.getMaxItemNum(targetDate) + 1

    logger.info("USER_NUM:%s", USER_NUM)
    logger.info("ITEM_NUM:%s", ITEM_NUM)

    infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    limit_size = 80

    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    # config.log_device_placement = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        initial_step = 0
        ckpt = tf.train.get_checkpoint_state(target_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = int(ckpt.model_checkpoint_path.rsplit("-", 1)[1])
            print("initial_step:", initial_step)

        unique_users = np.unique(total["user"])
        unique_items = np.unique(total["item"])

        new_users = [99]
        conn = dbio.getConnection()
        # dbio.truncate_user_item_rank_tmp()
        idx = 0
        for user in unique_users:

            if (user%thread_max == index):
                is_target_user, rec_cnt = dbio.is_target_user(conn, user, targetDate)
                if is_target_user is True:
                    pred_batch = sess.run(infer, feed_dict={user_batch: [user],
                                                            item_batch: unique_items})
                    new_item = np.zeros(rec_cnt)

                    # for i in range(0, rec_cnt):
                    #     new_item[i] =  unique_items[np.argsort(pred_batch)[len(unique_items) - i - 1]]

                    similar_indices = pred_batch.argsort()[:-(rec_cnt + 1):-1]
                    new_item = [unique_items[i] for i in similar_indices]
                    dbio.insert_user_item_rank_tmp(user, new_item, conn)

            if (idx % 1000 == 0):
                logger.info("%s processed", idx)

            if (idx>1000):
                break
            idx = idx + 1


        conn.close()
        # dbio.truncate_user_item_rank_a()
        # dbio.insert_user_item_rank_a(targetDate, limit_size)

    thread_result[index] = True
    insertData_end_time = time.time()
    logger.info("thread %s:  elapsed time:%s", index, insertData_end_time - insertData_start_time )

if __name__ == '__main__':
    total_start_time = time.time();
    df_train, df_test, df_total = get_data()

    logger.info("get_data() end")

    svd(df_total)

    total_end_time = time.time();
    logger.info("total elapsed time:%s",total_end_time - total_start_time )
    logger.info("Done!")

