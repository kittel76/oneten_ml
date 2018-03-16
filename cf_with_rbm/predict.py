from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from six import next
from tensorflow.core.framework import summary_pb2

from cf_with_rbm import dbio
from batch import dataio
from batch import ops
import shutil
import os
import logging.config
import logging
import cf_with_rbm.common as common


logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('cf_with_rbm')

# target_data_dir = "/data/www/oneten/dl_prd_rating/data_dir/" + args.targetDate
# target_train_dir = "/data/www/oneten/dl_prd_rating/train_dir/" + args.targetDate
# if not os.path.isdir(target_data_dir):
#     os.mkdir(target_data_dir)
#
#
# #Loading in the item dataset
# movies_df = pd.read_csv('/resources/data/ml-1m/movies.dat', sep='::', header=None)
# movies_df.head()


hiddenUnits = common.hiddenUnits
max_rating = common.max_rating

def main(targetDate):
    logger.info("targetDate:%s", targetDate)
    logger.info("hiddenUnits:%s", hiddenUnits)
    logger.info("max_rating:%s", max_rating)

    target_data_dir = "/data/www/oneten/data_cf_with_rbm/data_dir/" + args.targetDate
    target_train_dir = "/data/www/oneten/data_cf_with_rbm/train_dir/" + args.targetDate
    if not os.path.isdir(target_data_dir):
        os.mkdir(target_data_dir)

    #Loading in the item dataset
    items_df = pd.read_csv(target_data_dir + '/item.csv', sep=',', header=None)
    items_df.columns = ["idx_ot_prd_no"]
    items_df['List Index'] = items_df.index

    ratings_df = pd.read_csv(target_data_dir + '/rating.csv', sep=',', header=None)
    ratings_df.columns = ["idx_cust_no", "idx_ot_prd_no", "rating"]

    merged_df = items_df.merge(ratings_df, on="idx_ot_prd_no")

    userGroup = merged_df.groupby('idx_cust_no')


    amountOfUsedUsers = 1000000
    trX = []
    custMap = {}

    # For each user in the group

    idx = 0
    for userID, curUser in userGroup:
        # Create a temp that stores every movie's rating
        temp = [0] * len(items_df)
        # print("userID", userID)
        # For each movie in curUser's movie list
        for num, movie in curUser.iterrows():
            # Divide the rating by 5 and store it
            temp[movie['List Index']] = movie['rating'] / max_rating
        # Now add the list of ratings into the training list
        # print(temp)
        trX.append(temp)
        custMap[idx] = userID
        # Check to see if we finished adding in the amount of users for training
        if amountOfUsedUsers == 0:
            break
        amountOfUsedUsers -= 1
        idx = idx +1

    visibleUnits = len(items_df)
    vb = tf.placeholder("float", [visibleUnits])  # Number of unique movies
    hb = tf.placeholder("float", [hiddenUnits])  # Number of features we're going to learn
    W = tf.placeholder("float", [visibleUnits, hiddenUnits])

    print("visibleUnits", visibleUnits)

    # Phase 1: Input Processing
    v0 = tf.placeholder("float", [None, visibleUnits])
    _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
    # Phase 2: Reconstruction
    _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
    v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
    h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

    # Learning rate
    alpha = 1.0
    # Create the gradients
    w_pos_grad = tf.matmul(tf.transpose(v0), h0)
    w_neg_grad = tf.matmul(tf.transpose(v1), h1)
    # Calculate the Contrastive Divergence to maximize
    CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
    # Create methods to update the weights and biases
    update_w = W + alpha * CD
    update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
    update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

    err = v0 - v1
    err_sum = tf.reduce_mean(err * err)


    #Current weight
    cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
    #Current visible unit biases
    cur_vb = np.zeros([visibleUnits], np.float32)
    #Current hidden unit biases
    cur_hb = np.zeros([hiddenUnits], np.float32)
    #Previous weight
    prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
    #Previous visible unit biases
    prv_vb = np.zeros([visibleUnits], np.float32)
    #Previous hidden unit biases
    prv_hb = np.zeros([hiddenUnits], np.float32)

    cur_w_var = tf.Variable(cur_w, name="cur_w_var")
    cur_vb_var = tf.Variable(cur_vb, name="cur_vb_var")
    cur_hb_val = tf.Variable(cur_hb, name="cur_hb_val")


    global_step = tf.contrib.framework.get_or_create_global_step()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        logger.info("target_train_dir:%s", target_train_dir)
        ckpt = tf.train.get_checkpoint_state(target_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = int(ckpt.model_checkpoint_path.rsplit("-", 1)[1])
            logger.info("initial_step:%s", initial_step)
         # plt.show()

        # inputUser = [trX[75]]
        inputUser = trX


        cur_w_var = cur_w_var.assign(cur_w)
        cur_vb_var = cur_vb_var.assign(cur_vb)
        cur_hb_val = cur_hb_val.assign(cur_hb)

        rec = sess.run(v1, feed_dict={v0: trX, W: cur_w_var.eval(), vb: cur_vb_var.eval(), hb: cur_hb_val.eval()})

        logger.info("rec size:%s", len(rec))
        conn = dbio.getConnection()
        dbio.truncate_uesr_item_rank_tmp()
        targetUserCnt = 0
        for idx, val in enumerate(rec):
            # if idx >1:
            #     break
            scored_movies_df = items_df
            scored_movies_df["rating"] = val
            idxCustNo = custMap[idx]

            is_target_user, rec_cnt = dbio.is_target_user(conn, idxCustNo)
            item = scored_movies_df.sort_values(["rating"], ascending=False).head(rec_cnt)
            if len(item["idx_ot_prd_no"].values) > 0 and is_target_user == True:
                dbio.insert_uesr_item_rank_tmp(idxCustNo, item["idx_ot_prd_no"].values, conn)
                targetUserCnt = targetUserCnt + 1

            if idx%1000 == 0:
                 logger.info("%s processed!", idx)

        logger.info("totlaUser:%s", rec.__len__())
        logger.info("targetUserCnt:%s", targetUserCnt)
        conn.close()
        dbio.truncate_uesr_item_rank_b()
        dbio.insert_uesr_item_rank_b(80)


if __name__ == '__main__':
    import argparse
    from datetime import date

    parser = argparse.ArgumentParser()
    parser.add_argument("--targetDate", help="targetDate")
    args = parser.parse_args()
    if args.targetDate == None:
        args.targetDate = date.fromtimestamp(time.time() + 60 * 60 * 24).strftime('%Y%m%d')
    main(args.targetDate)