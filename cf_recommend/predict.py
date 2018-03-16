from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from cf_recommend import dataio
from cf_recommend import ops
from cf_recommend import dbio
import tensorflow as tf

np.random.seed(13575)

DIM = ops.DIM

DEVICE = "/cpu:0"


def get_data():
    df = dataio.read_process("/data/www/oneten/dl_cf_recommend/click.csv", sep=",")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    return df

def get_unique_item():
    return np.loadtxt("/data/www/oneten/dl_cf_recommend/data_dir/item.csv", delimiter=',', dtype=np.int32)


def get_unique_user():
    return np.loadtxt("/data/www/oneten/dl_cf_recommend/data_dir/user.csv", delimiter=',', dtype=np.int32)

def getItemRank(target_users, limit_size):


    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])


    USER_NUM = dbio.getMaxUserNum() + 1
    ITEM_NUM = dbio.getMaxItemNum() + 1


    print("USER_NUM:", USER_NUM)
    print("ITEM_NUM:", ITEM_NUM)

    if limit_size > (ITEM_NUM - 1):
        limit_size = (ITEM_NUM - 1)

    infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    print("step3")
    with tf.Session() as sess:
        sess.run(init_op)
        initial_step = 0
        ckpt = tf.train.get_checkpoint_state("/data/www/oneten/dl_cf_recommend/train_dir")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = int(ckpt.model_checkpoint_path.rsplit("-", 1)[1])
            print("initial_step:", initial_step)

        print("unique_items befefore")
        unique_items = get_unique_item()
        print("unique_items after")

        for user in target_users:
            print("user:", user)
            pred_batch = sess.run(infer, feed_dict={user_batch: [user],
                                                    item_batch: unique_items})
            new_item = np.zeros(len(pred_batch))
            print("pred_batch", pred_batch)
            print("step1")
            # for i in range(0, len(new_item)):
            #     new_item[len(new_item) - 1 - i] = unique_items[np.argsort(pred_batch)[i]]

            print("11")
            for i in range(0, limit_size):
                new_item[i] = unique_items[np.argsort(pred_batch)[len(unique_items)-i-1]]
            print("aa:", unique_items[np.argsort(pred_batch)[len(new_item)-1]])
            print("step2")
            np_new_item = np.asarray(new_item).astype(int)
            print("22")


    return np_new_item[0:limit_size]

if __name__ == '__main__':
    target_users = [99]
    limit_size = 4
    item_rank = getItemRank(target_users, limit_size)
    print("item_rank", item_rank)

