from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from collections import deque

import numpy as np
import tensorflow as tf
from six import next
from tensorflow.core.framework import summary_pb2

from getin.cf_recommend import dbio
from getin.cf_recommend import dataio
from getin.cf_recommend import ops
from getin.cf_recommend import common
import shutil
import os
import logging.config
import logging



from datetime import datetime
np.random.seed(13575)

import argparse
from datetime import date
parser = argparse.ArgumentParser()
parser.add_argument("--targetDate", help="targetDate")
args = parser.parse_args()
if args.targetDate == None:
    args.targetDate = date.fromtimestamp(time.time() + 60 * 60 * 24).strftime('%Y%m%d')

target_data_dir = common.DATA_HOME + "/data_dir/" + args.targetDate
target_train_dir = common.DATA_HOME + "/train_dir/" + args.targetDate
if not os.path.isdir(target_data_dir):
    os.mkdir(target_data_dir)

BATCH_SIZE = common.BATCH_SIZE
DIM = ops.DIM
EPOCH_MAX = common.EPOCH_MAX
DEVICE = "/gpu:0"
if os.getenv("pythonAppType", "") == "local":
    DEVICE = "/cpu:0"


logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('cf_recommend')

logger.info("args.targetDate:%s", args.targetDate)


def clip(x):
    return np.clip(x, 0, 15)

def make_scalar_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def get_data():
    df = dataio.read_process(target_data_dir + "/click.csv", sep=",")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test, df


def svd(train, test, total):

    train_len = len(train)
    samples_per_batch = len(train) // BATCH_SIZE

    logger.info("samples_per_batch:%s", samples_per_batch)

    logger.info("samples_per_batch:%s", samples_per_batch)

    # USER_NUM = dbio.getMaxUserNum()
    USER_NUM = dbio.getMaxUserNum(args.targetDate) + 1
    ITEM_NUM = dbio.getMaxItemNum(args.targetDate) + 1


    logger.info("USER_NUM:%s", USER_NUM)
    logger.info("ITEM_NUM:%s", ITEM_NUM)

    iter_train = dataio.ShuffleIterator([train["user"],
                                         train["item"],
                                         train["rate"]],
                                        batch_size=BATCH_SIZE)


    iter_test = dataio.OneEpochIterator([test["user"],
                                         test["item"],
                                         test["rate"]],
                                        batch_size=-1)

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])



    infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    init_op = tf.global_variables_initializer()

    logger.info("samples_per_batch:%s", samples_per_batch)

    shutil.rmtree(target_train_dir, ignore_errors=True)
    os.mkdir(target_train_dir)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.log_device_placement = True
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        logger.info("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()

        logger.info("EPOCH_MAX * samples_per_batch:%s", EPOCH_MAX * samples_per_batch)
        logger.info("samples_per_batch:%s", samples_per_batch)

        for i in range(EPOCH_MAX * samples_per_batch):

            users, items, rates = next(iter_train)

            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates})


            errors.append(np.power(pred_batch - rates, 2))
            # if i % samples_per_batch == 0:
            if i % 1000 == 1:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items})
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                logger.info("{} step:{:d} error {:3d} {:f} {:f} {:f}(s)".format(datetime.now(), i, i // samples_per_batch, train_err, test_err, end - start))
                train_err_summary = make_scalar_summary("training_error", train_err)
                test_err_summary = make_scalar_summary("test_error", test_err)
                summary_writer.add_summary(train_err_summary, i)
                summary_writer.add_summary(test_err_summary, i)
                start = end


        saver.save(sess, target_train_dir + "/item-rating", global_step=i)



if __name__ == '__main__':
    df_train, df_test, df_total = get_data()
    total_start_time = time.time();
    svd(df_total, df_test, df_total)
    total_end_time = time.time();

    # print("df_total,", df_total)
    logger.info("total elapsed time:%s",total_end_time - total_start_time )
    logger.info("Done!")

