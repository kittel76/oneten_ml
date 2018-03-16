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
import sys, gc
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


    global thread_result
    global limit_size
    global USER_NUM, ITEM_NUM

    global user_batch, item_batch
    global infer, regularizer
    global unique_users, unique_items


    unique_users = np.unique(total["user"])
    unique_items = np.unique(total["item"])

    limit_size = 150
    USER_NUM = dbio.getMaxUserNum(targetDate) + 1
    ITEM_NUM = dbio.getMaxItemNum(targetDate) + 1

    logger.info("USER_NUM:%s", USER_NUM)
    logger.info("ITEM_NUM:%s", ITEM_NUM)


    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")



    infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init_op)

    ckpt = tf.train.get_checkpoint_state(target_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    dbio.truncate_user_item_rank_tmp()

    import threading

    thread_size = 4
    thread_result = np.zeros(thread_size)

    for i in range(thread_size):
        threading.Thread(target=insertData, args=(i, thread_size, sess)).start()

    while 1:
        time.sleep(5)
        logger.info("check thread:[%s]",  thread_result )
        if(thread_result.all() == 1 ):
            logger.info("0 thread is end")
            break


    dbio.truncate_user_item_rank_a()
    logger.info("truncate")
    dbio.insert_user_item_rank_a(targetDate, limit_size)
    logger.info("insert")


def insertData(index, thread_max, sess):
    insertData_start_time = time.time()



    conn = dbio.getConnectionByOption(autocommit=False, threaded=True)
    connInsert = dbio.getConnectionByOption(autocommit=True, threaded=True)
    print("conn.autocommit", conn.autocommit)
    # dbio.truncate_user_item_rank_tmp()
    idx = 0
    for user in unique_users:

        if (user%thread_max == index):
            # is_target_user, rec_cnt = dbio.is_target_user_thread(user)

            is_target_user, rec_cnt = dbio.is_target_user(conn, user, target_date=targetDate)
            # is_target_user = True
            rec_cnt = limit_size
            if is_target_user is True:
                pred_batch = sess.run(infer, feed_dict={user_batch: [user],
                                                        item_batch: unique_items})
                new_item = np.zeros(rec_cnt)

                similar_indices = pred_batch.argsort()[:-(rec_cnt + 1):-1]
                new_item = [unique_items[i] for i in similar_indices]
                dbio.insert_user_item_rank_tmp_nocommit(user, new_item, connInsert)


        if (idx % 1000 == 0):
            logger.info("%s processed", idx)


        idx = idx + 1


    conn.close()
    connInsert.close()
    thread_result[index] = 1
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

