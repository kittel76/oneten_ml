# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import magic
from PIL import Image
import tensorflow as tf
from flask import Flask
from flask import request
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from web import image_util
import urllib
import os
from datetime import datetime
from img_category_process3 import category_util3
import pandas as pd
from title_category_classify import predict
from title_category_classify import common

serverCategory = 'dl:9001'
serverStyle = 'dl:9002'
serverColor = 'dl:9003'
serverPattern = 'dl:9004'


if os.getenv("pythonAppType", "") == "local":
    serverCategory = 'dl:9001'
    serverStyle = 'dl:9002'
    serverColor = 'dl:9003'
    serverPattern = 'dl:9004'


tf.app.flags.DEFINE_string('server_category', serverCategory,
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('server_style', serverStyle,
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('server_color', serverColor,
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('server_pattern', serverPattern,
                           'PredictionService host:port')

FLAGS = tf.app.flags.FLAGS

app = Flask(__name__)

sess = tf.Session()




@app.route('/api/getCategoryInfo3')
def getCategoryInfo3 ():

    start = datetime.now()

    img_url = request.args.get('img_url')
    rand = random.randint(1, 100000)
    tmpFileName = "/tmp/img_" + rand.__str__() + ".jpg"
    tmpResizeFileName = "/tmp/img_resize_" + rand.__str__() + ".jpg"


    tmpFileName, _ = urllib.urlretrieve(img_url, tmpFileName)


    type = magic.from_file(tmpFileName).split()[0].lower()
    if (type == "gif"):
        image_util.gifToJpeg(tmpFileName, tmpFileName)

    size = 128, 128
    image_util.resize_and_crop(tmpFileName, tmpResizeFileName, size, crop_type="middle")

    with tf.Session() as sess:
        images = image_util.get_process_image_with_sess(sess, tmpResizeFileName, 96, 96)
        images_ = sess.run(images)



    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_category.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(images_, shape=[128,96,96,3]))

    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch = result.result().outputs['class'].int64_val[0]


    unique_labels = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_category_process3/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels.__len__()):
        dictionary[idx] = unique_labels[idx]

    result = "{\"cate_no\":" + str(dictionary[result_batch]).split("_")[0] + ", \"cate_nm\":\"" + str(dictionary[result_batch]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"


    return result






@app.route('/api/getStyleInfo2')
def getStyleInfo2 ():

    img_url = request.args.get('img_url')
    rand = random.randint(1, 100000)
    tmpFileName = "/tmp/img_" + rand.__str__() + ".jpg"
    tmpResizeFileName = "/tmp/img_resize_" + rand.__str__() + ".jpg"

    tmpFileName, _ = urllib.urlretrieve(img_url, tmpFileName)
    type = magic.from_file(tmpFileName).split()[0].lower()
    if (type == "gif"):
        image_util.gifToJpeg(tmpFileName, tmpFileName)

    size_64 = 48, 48
    image_util.resize_and_crop(tmpFileName, tmpResizeFileName, size_64, crop_type="middle")
    images_ = image_util.get_process_image_real(tmpResizeFileName)


    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_style.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(images_, shape=[128,48,48,3]))

    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch = result.result().outputs['class'].int64_val[0]

    unique_labels = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_style_process2/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels.__len__()):
        dictionary[idx] = unique_labels[idx]

    result = "{\"attr_no\":" + str(dictionary[result_batch]).split("_")[0] + ", \"style_nm\":\"" + str(dictionary[result_batch]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"

    return result


@app.route('/api/getColorInfo2')
def getColorInfo2 ():

    img_url = request.args.get('img_url')
    rand = random.randint(1, 100000)
    tmpFileName = "/tmp/img_" + rand.__str__() + ".jpg"
    tmpResizeFileName = "/tmp/img_resize_" + rand.__str__() + ".jpg"

    tmpFileName, _ = urllib.urlretrieve(img_url, tmpFileName)
    type = magic.from_file(tmpFileName).split()[0].lower()
    if (type == "gif"):
        image_util.gifToJpeg(tmpFileName, tmpFileName)

    size_64 = 48, 48
    image_util.resize_and_crop(tmpFileName, tmpResizeFileName, size_64, crop_type="middle")
    images_ = image_util.get_process_image_real(tmpResizeFileName)


    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_color.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(images_, shape=[128,48,48,3]))

    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch = result.result().outputs['class'].int64_val[0]


    print("step4", datetime.now())

    unique_labels = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_color_process2/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels.__len__()):
        dictionary[idx] = unique_labels[idx]

    result = "{\"attr_no\":" + str(dictionary[result_batch]).split("_")[0] + ", \"style_nm\":\"" + str(dictionary[result_batch]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"

    print("step5", datetime.now())

    os.remove(tmpFileName)
    os.remove(tmpResizeFileName)

    return result


@app.route('/api/getPatternInfo2')
def getPatternInfo2 ():

    img_url = request.args.get('img_url')
    rand = random.randint(1, 100000)
    tmpFileName = "/tmp/img_" + rand.__str__() + ".jpg"
    tmpResizeFileName = "/tmp/img_resize_" + rand.__str__() + ".jpg"

    tmpFileName, _ = urllib.urlretrieve(img_url, tmpFileName)
    type = magic.from_file(tmpFileName).split()[0].lower()
    if (type == "gif"):
        image_util.gifToJpeg(tmpFileName, tmpFileName)

    size_64 = 48, 48
    image_util.resize_and_crop(tmpFileName, tmpResizeFileName, size_64, crop_type="middle")
    images_ = image_util.get_process_image_real(tmpResizeFileName)


    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_pattern.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(images_, shape=[128,48,48,3]))

    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch = result.result().outputs['class'].int64_val[0]


    print("step4", datetime.now())

    unique_labels = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_pattern_process2/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels.__len__()):
        dictionary[idx] = unique_labels[idx]

    result = "{\"attr_no\":" + str(dictionary[result_batch]).split("_")[0] + ", \"pattern_nm\":\"" + str(dictionary[result_batch]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"

    print("step5", datetime.now())

    os.remove(tmpFileName)
    os.remove(tmpResizeFileName)

    return result

@app.route('/api/getImgExtraInfo')
def getImgExtraInfo ():

    img_url = request.args.get('img_url')
    prd_nm = request.args.get("prd_nm")
    if prd_nm is None:
        prd_nm = ''
    prd_nm = prd_nm.encode('utf-8')
    rand = random.randint(1, 100000)

    tmpFileName = "/tmp/img_" + rand.__str__() + ".jpg"
    tmpResizeFileNameFor128 = "/tmp/img_resize_" + rand.__str__() + "_128.jpg"
    tmpResizeFileNameFor64 = "/tmp/img_resize_" + rand.__str__() + "_64.jpg"
    tmpCropFileNameFor96 = "/tmp/img_resize_" + rand.__str__() + "_96.jpg"
    tmpCropFileNameFor48 = "/tmp/img_resize_" + rand.__str__() + "_48.jpg"


    tmpFileName, _ = urllib.urlretrieve(img_url, tmpFileName)


    type = magic.from_file(tmpFileName).split()[0].lower()
    if (type == "gif"):
        image_util.gifToJpeg(tmpFileName, tmpFileName)

    size_128 = 128, 128
    size_96 = 96, 96
    image_util.resize_and_crop(tmpFileName, tmpResizeFileNameFor128, size_128, crop_type="middle")
    image_util.crop(tmpResizeFileNameFor128, tmpCropFileNameFor96, size_96)


    size_64 = 64, 64
    size_48 = 48, 48
    image_util.resize_and_crop(tmpFileName, tmpResizeFileNameFor64, size_64, crop_type="middle")
    image_util.crop(tmpResizeFileNameFor64, tmpCropFileNameFor48, size_48)



    imagesCategory_ = image_util.get_process_image_real(tmpCropFileNameFor96)
    imagesStyle_ = image_util.get_process_image_real(tmpCropFileNameFor48)
    imagesColor_ = image_util.get_process_image_real(tmpCropFileNameFor48)


    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_category.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(imagesCategory_, shape=[128,96,96,3]))
    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch_category = result.result().outputs['class'].int64_val[0]


    unique_labels_category = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_category_process3/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels_category.__len__()):
        dictionary[idx] = unique_labels_category[idx]

    resultCategory = "{\"cate_no\":" + str(dictionary[result_batch_category]).split("_")[0] + ", \"cate_nm\":\"" + str(dictionary[result_batch_category]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"


    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_style.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(imagesStyle_, shape=[128,48,48,3]))
    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch_style = result.result().outputs['class'].int64_val[0]


    unique_labels_style = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_style_process2/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels_style.__len__()):
        dictionary[idx] = unique_labels_style[idx]

    resultStyle = "{\"attr_no\":" + str(dictionary[result_batch_style]).split("_")[0] + ", \"style_nm\":\"" + str(dictionary[result_batch_style]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"



    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_color.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(imagesColor_, shape=[128,48,48,3]))
    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch_color = result.result().outputs['class'].int64_val[0]

    unique_labels_color = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_color_process2/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels_color.__len__()):
        dictionary[idx] = unique_labels_color[idx]

    resultColor = "{\"attr_no\":" + str(dictionary[result_batch_color]).split("_")[0] + ", \"style_nm\":\"" + str(dictionary[result_batch_color]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"

    os.remove(tmpFileName);
    os.remove(tmpResizeFileNameFor128);
    os.remove(tmpResizeFileNameFor64);
    os.remove(tmpCropFileNameFor96);
    os.remove(tmpCropFileNameFor48);


    result_tot =  "{" \
                   + " \"category\":" + resultCategory \
                   + " ,\"category_from_prd_nm\":" + getCategoryStrFromPrdNm(prd_nm) \
                   + " ,\"style\":" + resultStyle \
                   + " ,\"color\":" + resultColor \
                   + "}"

    return result_tot





@app.route('/api/getImgExtraInfoV2')
def getImgExtraInfoV2 ():

    img_url = request.args.get('img_url')
    prd_nm = request.args.get("prd_nm")
    if prd_nm is None:
        prd_nm = ''
    prd_nm = prd_nm.encode('utf-8')
    rand = random.randint(1, 100000)

    tmpFileName = "/tmp/img_" + rand.__str__() + ".jpg"
    tmpResizeFileNameFor128 = "/tmp/img_resize_" + rand.__str__() + "_128.jpg"
    tmpResizeFileNameFor64 = "/tmp/img_resize_" + rand.__str__() + "_64.jpg"
    tmpCropFileNameFor96 = "/tmp/img_resize_" + rand.__str__() + "_96.jpg"
    tmpCropFileNameFor48 = "/tmp/img_resize_" + rand.__str__() + "_48.jpg"


    tmpFileName, _ = urllib.urlretrieve(img_url, tmpFileName)


    type = magic.from_file(tmpFileName).split()[0].lower()
    if (type == "gif"):
        image_util.gifToJpeg(tmpFileName, tmpFileName)

    size_128 = 128, 128
    size_96 = 96, 96
    image_util.resize_and_crop(tmpFileName, tmpResizeFileNameFor128, size_128, crop_type="middle")
    image_util.crop(tmpResizeFileNameFor128, tmpCropFileNameFor96, size_96)


    size_64 = 64, 64
    size_48 = 48, 48
    image_util.resize_and_crop(tmpFileName, tmpResizeFileNameFor64, size_64, crop_type="middle")
    image_util.crop(tmpResizeFileNameFor64, tmpCropFileNameFor48, size_48)



    imagesCategory_ = image_util.get_process_image_real(tmpCropFileNameFor96)
    imagesStyle_ = image_util.get_process_image_real(tmpCropFileNameFor48)
    imagesColor_ = image_util.get_process_image_real(tmpCropFileNameFor48)
    imagesPattern_ = image_util.get_process_image_real(tmpCropFileNameFor48)


    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_category.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(imagesCategory_, shape=[128,96,96,3]))
    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch_category = result.result().outputs['class'].int64_val[0]


    unique_labels_category = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_category_process3/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels_category.__len__()):
        dictionary[idx] = unique_labels_category[idx]

    resultCategory = "{\"cate_no\":" + str(dictionary[result_batch_category]).split("_")[0] + ", \"cate_nm\":\"" + str(dictionary[result_batch_category]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"


    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_style.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(imagesStyle_, shape=[128,48,48,3]))
    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch_style = result.result().outputs['class'].int64_val[0]


    unique_labels_style = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_style_process2/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels_style.__len__()):
        dictionary[idx] = unique_labels_style[idx]

    resultStyle = "{\"attr_no\":" + str(dictionary[result_batch_style]).split("_")[0] + ", \"style_nm\":\"" + str(dictionary[result_batch_style]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"


    # color start
    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_color.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(imagesColor_, shape=[128,48,48,3]))
    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch_color = result.result().outputs['class'].int64_val[0]

    unique_labels_color = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_color_process2/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels_color.__len__()):
        dictionary[idx] = unique_labels_color[idx]

    resultColor = "{\"attr_no\":" + str(dictionary[result_batch_color]).split("_")[0] + ", \"color_nm\":\"" + str(dictionary[result_batch_color]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"

    # color end


    # pattern start
    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_pattern.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(imagesPattern_, shape=[128,48,48,3]))
    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch_pattern = result.result().outputs['class'].int64_val[0]

    unique_labels_pattern = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_pattern_process2/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels_pattern.__len__()):
        dictionary[idx] = unique_labels_pattern[idx]

    resultPattern = "{\"attr_no\":" + str(dictionary[result_batch_pattern]).split("_")[0] + ", \"pattern_nm\":\"" + str(dictionary[result_batch_pattern]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"

    # pattern end


    os.remove(tmpFileName);
    os.remove(tmpResizeFileNameFor128);
    os.remove(tmpResizeFileNameFor64);
    os.remove(tmpCropFileNameFor96);
    os.remove(tmpCropFileNameFor48);

    if prd_nm != '':
        resultCategory = getCategoryStrFromPrdNm(prd_nm)


    result_tot =  "{" \
                   + " \"category\":" + resultCategory \
                   + " ,\"style\":" + resultStyle \
                   + " ,\"color\":" + resultColor \
                   + " ,\"pattern\":" + resultPattern \
                  + "}"

    return result_tot

# @app.route('/api/getImgExtraInfoV2')
def getImgExtraInfoV2Tmp ():

    img_url = request.args.get('img_url')
    prd_nm = request.args.get("prd_nm")
    if prd_nm is None:
        prd_nm = ''
    prd_nm = prd_nm.encode('utf-8')
    rand = random.randint(1, 100000)


    if prd_nm != '':
        resultCategory = getCategoryStrFromPrdNm(prd_nm)


    result_tot =  "{" \
                   + " \"category\":" + resultCategory \
                   + " ,\"style\":" + "\"\"" \
                   + " ,\"color\":" + "\"\"" \
                   + "}"

    return result_tot




@app.route('/api/getImgExtraInfoCompare')
def getImgExtraInfoCompare ():

    img_url = request.args.get('img_url')
    rand = random.randint(1, 100000)

    tmpFileName = "/tmp/img_" + rand.__str__() + ".jpg"
    tmpResizeFileNameFor128 = "/tmp/img_resize_" + rand.__str__() + "_128.jpg"
    tmpResizeFileNameFor64 = "/tmp/img_resize_" + rand.__str__() + "_64.jpg"
    tmpCropFileNameFor96 = "/tmp/img_resize_" + rand.__str__() + "_96.jpg"
    tmpCropFileNameFor48 = "/tmp/img_resize_" + rand.__str__() + "_48.jpg"


    tmpFileName, _ = urllib.urlretrieve(img_url, tmpFileName)


    type = magic.from_file(tmpFileName).split()[0].lower()
    if (type == "gif"):
        image_util.gifToJpeg(tmpFileName, tmpFileName)

    size_128 = 128, 128
    size_96 = 96, 96
    image_util.resize_and_crop(tmpFileName, tmpResizeFileNameFor128, size_128, crop_type="middle")
    image_util.crop(tmpResizeFileNameFor128, tmpCropFileNameFor96, size_96)


    size_64 = 64, 64
    size_48 = 48, 48
    image_util.resize_and_crop(tmpFileName, tmpResizeFileNameFor64, size_64, crop_type="middle")
    image_util.crop(tmpResizeFileNameFor64, tmpCropFileNameFor48, size_48)



    imagesCategory_ = image_util.get_process_image_real(tmpCropFileNameFor96)
    imagesStyle_ = image_util.get_process_image_real(tmpCropFileNameFor48)
    imagesColor_ = image_util.get_process_image_real(tmpCropFileNameFor48)


    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_category.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(imagesCategory_, shape=[128,96,96,3]))
    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch_category = result.result().outputs['class'].int64_val[0]


    unique_labels_category = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_category_process3/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels_category.__len__()):
        dictionary[idx] = unique_labels_category[idx]

    resultCategory = "{\"cate_no\":" + str(dictionary[result_batch_category]).split("_")[0] + ", \"cate_nm\":\"" + str(dictionary[result_batch_category]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"


    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_style.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(imagesStyle_, shape=[128,48,48,3]))
    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch_style = result.result().outputs['class'].int64_val[0]


    unique_labels_style = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_style_process2/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels_style.__len__()):
        dictionary[idx] = unique_labels_style[idx]

    resultStyle = "{\"attr_no\":" + str(dictionary[result_batch_style]).split("_")[0] + ", \"style_nm\":\"" + str(dictionary[result_batch_style]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"



    req = predict_pb2.PredictRequest()
    host, port = FLAGS.server_color.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    req.model_spec.name = 'cifar10'
    req.model_spec.signature_name = 'predict_images'
    req.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(imagesColor_, shape=[128,48,48,3]))
    result = stub.Predict.future(req, 60.0)  # 60 secs timeout
    result_batch_color = result.result().outputs['class'].int64_val[0]

    unique_labels_color = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_color_process2/data_dir/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels_color.__len__()):
        dictionary[idx] = unique_labels_color[idx]

    resultColor = "{\"attr_no\":" + str(dictionary[result_batch_color]).split("_")[0] + ", \"style_nm\":\"" + str(dictionary[result_batch_color]).split("_")[1]+ "\", \"img_url\":\"" + str(img_url) + "\"" +  "}"

    os.remove(tmpFileName);
    os.remove(tmpResizeFileNameFor128);
    os.remove(tmpResizeFileNameFor64);

    result_tot =  "{" \
                   + " \"category\":" + resultCategory \
                   + " ,\"category2\":" + getCategoryStr(img_url) \
                   + " ,\"style\":" + resultStyle \
                   + " ,\"color\":" + resultColor \
                   + "}"

    return result_tot



@app.route('/api/getCategoryInfo3_fromFile')
def getCategoryInfo3_fromFile ():
    img_url = request.args.get('img_url')
    result = getCategoryStr(img_url)


    return result


def getCategoryStr(img_url):
    img_url = request.args.get('img_url')
    rand = random.randint(1, 100000)
    tmpFileName = "/tmp/img_" + rand.__str__() + ".jpg"
    tmpResizeFileName = "/tmp/img_resize_" + rand.__str__() + ".jpg"

    tmpFileName, _ = urllib.urlretrieve(img_url, tmpFileName)
    type = magic.from_file(tmpFileName).split()[0].lower()
    if (type == "gif"):
        image_util.gifToJpeg(tmpFileName, tmpFileName)

    size = 128, 128
    image_util.resize_and_crop(tmpFileName, tmpResizeFileName, size, crop_type="middle")
    ranking = category_util3.getCategory(tmpResizeFileName)

    unique_labels = [l.strip() for l in
                     tf.gfile.FastGFile('/data/www/oneten/dl_img_category_process3/data_dir/labels.txt',
                                        'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels.__len__()):
        dictionary[idx] = unique_labels[idx]

    result = "{\"cate_no\":" + str(dictionary[ranking[0]]).split("_")[0] + ", \"cate_nm\":\"" + \
             str(dictionary[ranking[0]]).split("_")[1] + "\", \"img_url\":\"" + str(img_url) + "\"" + "}"

    return result

df = pd.read_csv(common.DATA_DIR_FOR_SERVICE +"/data.csv", header=1, sep='###', names=["prd_nm", "idx"])
x_train = pd.Series(df["prd_nm"])
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(common.MAX_DOCUMENT_LENGTH)
x_transform_train = vocab_processor.fit_transform(x_train)

@app.route('/api/getCategoryStrFromPrdNm')
def getCategoryStrFromPrdNm():

    # img_url = request.args.get('img_url')
    prd_nm = request.args.get("prd_nm").encode('utf-8')

    return getCategoryStrFromPrdNm(prd_nm)

def getCategoryStrFromPrdNm(prd_nm):

    # img_url = request.args.get('img_url')

    x_test = [prd_nm]

    print("prd_nm", prd_nm)
    print("x_test", x_test)
    y_predicted = predict.getPredict(prd_nm=prd_nm, vocab_processor=vocab_processor)
    result_batch = y_predicted[0]-1

    print("result_batch", result_batch)

    unique_labels = [l.strip() for l in
                     tf.gfile.FastGFile(common.DATA_DIR_FOR_SERVICE +  '/labels.txt', 'r').readlines()]
    dictionary = dict()
    for idx in range(unique_labels.__len__()):
        dictionary[idx] = unique_labels[idx]

    result = "{\"cate_no\":" + str(dictionary[result_batch]).split("_")[0] + ", \"cate_nm\":\"" + str(dictionary[result_batch]).split("_")[1]+ "\", \"prd_nm\":\"" + prd_nm + "\"" +  "}"

    return result





if __name__ == '__main__':

    app.run("0.0.0.0", "8000")

