from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# epochs = 10
epochs = 25
batchsize = 1000
hiddenUnits = common.hiddenUnits
max_rating = common.max_rating

def main(targetDate):
    logger.info("targetDate:%s", targetDate)

    target_data_dir = "/data/www/oneten/data_cf_with_rbm/data_dir/" + args.targetDate
    target_train_dir = "/data/www/oneten/data_cf_with_rbm/train_dir/" + args.targetDate
    if not os.path.isdir(target_data_dir):
        os.mkdir(target_data_dir)
    shutil.rmtree(target_train_dir, ignore_errors=True)
    os.mkdir(target_train_dir)

    #Loading in the item dataset
    items_df = pd.read_csv(target_data_dir + '/item.csv', sep=',', header=None)
    items_df.columns = ["idx_ot_prd_no"]
    items_df['List Index'] = items_df.index

    logger.info("read_cvs: %s", target_data_dir + '/item.csv')

    ratings_df = pd.read_csv(target_data_dir + '/rating.csv', sep=',', header=None)
    ratings_df.columns = ["idx_cust_no", "idx_ot_prd_no", "rating"]

    logger.info("read_cvs: %s", target_data_dir + '/rating.csv')

    merged_df = items_df.merge(ratings_df, on="idx_ot_prd_no")

    logger.info("merged_df completed")

    userGroup = merged_df.groupby('idx_cust_no')

    logger.info("userGroup groupby idx_cust_no")


    trX = []
    custMap = {}
    # amountOfUsedUsers = amountOfUsedUsers
    # For each user in the group
    amountOfUsedUsers = 1000000
    idx = 0
    for userID, curUser in userGroup:
        # Create a temp that stores every movie's rating
        temp = [0] * len(items_df)
        # For each movie in curUser's movie list
        for num, movie in curUser.iterrows():
            # Divide the rating by 5 and store it
            temp[movie['List Index']] = movie['rating'] / max_rating
        # Now add the list of ratings into the training list
        # print(temp)
        trX.append(temp)
        custMap[idx] = userID
        # Check to see if we finished adding in the amount of users for training
        # if amountOfUsedUsers == 0:
        #     break
        # amountOfUsedUsers -= 1
        amountOfUsedUsers = amountOfUsedUsers -1
        idx = idx +1

    logger.info("trX completed")


    visibleUnits = len(items_df)
    vb = tf.placeholder("float", [visibleUnits])  # Number of unique movies
    hb = tf.placeholder("float", [hiddenUnits])  # Number of features we're going to learn
    W = tf.placeholder("float", [visibleUnits, hiddenUnits])


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

    global_step = tf.contrib.framework.get_or_create_global_step()

    cur_w_var = tf.Variable(cur_w, name="cur_w_var")
    cur_vb_var = tf.Variable(cur_vb, name="cur_vb_var")
    cur_hb_val = tf.Variable(cur_hb, name="cur_hb_val")

    saver = tf.train.Saver()

    logger.info("session start")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.log_device_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        errors = []
        step = 0
        logger.info("len(trX):%s", len(trX))
        for i in range(epochs):
            for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
                logger.info("epoch:%s start:%s, end:%s", i, start, end)
                batch = trX[start:end]
                cur_w = sess.run( update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
                prv_w = cur_w
                prv_vb = cur_vb
                prv_hb = cur_hb
            errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
            logger.info ("epoch:%s,  epochs errors: %s", i, errors[-1])

        cur_w_var = cur_w_var.assign(cur_w)
        cur_vb_var = cur_vb_var.assign(cur_vb)
        cur_hb_val = cur_hb_val.assign(cur_hb)


        saver.save(sess, target_train_dir + "/item-rating", global_step=epochs * batchsize)
        # print("prv_w_var", prv_w_var.eval())

        # plt.plot(errors)
        # plt.ylabel('Error')
        # plt.xlabel('Epoch')
        # plt.show()




if __name__ == '__main__':
    import argparse
    from datetime import date

    logger.info("start")
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--targetDate", help="targetDate")
    args = parser.parse_args()
    if args.targetDate == None:
        args.targetDate = date.fromtimestamp(time.time() + 60 * 60 * 24).strftime('%Y%m%d')
    main(args.targetDate)

    end = time.time() - start

    logger.info("end")
    print("elapsed time: %s", end)