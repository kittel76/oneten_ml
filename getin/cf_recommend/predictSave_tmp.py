from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from collections import deque

import numpy as np
import tensorflow as tf


from getin.cf_recommend import dataio
from getin.cf_recommend import common
from getin.cf_recommend import ops
from getin.cf_recommend import dbio

np.random.seed(13575)

DIM = ops.DIM
DEVICE = "/cpu:0"


import argparse
from datetime import date
parser = argparse.ArgumentParser()
parser.add_argument("--targetDate", help="targetDate")
args = parser.parse_args()
if args.targetDate == None:
    args.targetDate = date.fromtimestamp(time.time() + 60 * 60 * 24).strftime('%Y%m%d')


target_data_dir = common.DATA_HOME + "/data_dir/" + args.targetDate
target_train_dir = common.DATA_HOME + "/train_dir/" + args.targetDate


import logging.config
import logging
logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('batch')

logger.info ("args.targetDate:%s", args.targetDate)

def get_data():
    df = dataio.read_process(target_data_dir + "/click.csv", sep=",")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test, df


def svd(total):


    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])


    USER_NUM = dbio.getMaxUserNum(args.targetDate) + 1
    ITEM_NUM = dbio.getMaxItemNum(args.targetDate) + 1


    logger.info("USER_NUM:%s", USER_NUM)
    logger.info("ITEM_NUM:%s", ITEM_NUM)

    infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    limit_size = 80

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.log_device_placement = True
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

        conn = dbio.getConnection()
        dbio.truncate_user_item_rank_tmp()
        for user in unique_users:
            is_target_user, rec_cnt = dbio.is_target_user(conn, user, args.targetDate)
            if is_target_user is False:
                continue
            pred_batch = sess.run(infer, feed_dict={user_batch: [user],
                                                    item_batch: unique_items})
            new_item = np.zeros(rec_cnt)

            for i in range(0, rec_cnt):
                new_item[i] =  unique_items[np.argsort(pred_batch)[len(unique_items) - i - 1]]
            dbio.insert_user_item_rank_tmp(user, new_item, conn, args.targetDate)
        conn.close()
        dbio.truncate_uesr_item_rank()
        dbio.insert_user_item_rank(args.targetDate, limit_size)

if __name__ == '__main__':
    total_start_time = time.time();
    df_train, df_test, df_total = get_data()
    svd(df_total)
    total_end_time = time.time();

    logger.info("total elapsed time:%s",total_end_time - total_start_time )
    logger.info("Done!")

