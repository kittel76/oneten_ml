# coding=utf-8
#!/usr/bin/env python2.7

'''
Send JPEG image to tensorflow_model_server loaded with GAN model.

Hint: the code has been compiled together with TensorFlow serving
and not locally. The client is called in the TensorFlow Docker container
'''

from __future__ import print_function

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf
from web import image_util
import os

# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import numpy as np

server = 'dl.wishlink.info:9002'
if os.getenv("pythonAppType", "") == "local":
    server = 'dl.wishlink.info:9002'

tf.app.flags.DEFINE_string('server', server,
                           'PredictionService host:port')

FLAGS = tf.app.flags.FLAGS

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image




def get_process_image(filename):


  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()



  coder = ImageCoder()
  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data) # numpy.array

  # 텐서로 변환
  reshaped_image = tf.cast(image, tf.float32)
  print("reshaped_image", reshaped_image)

  height = 96
  width = 96

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)

  print("resized_image", resized_image)
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, 3])
  print("float_image", float_image)

  images = tf.train.batch(
    [float_image],
    batch_size=128)

  return images


def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)


    filename = "./skirt.jpg"

    sess = tf.Session()
    images = image_util.get_process_image_sess(filename, 48, 48)
    images_ = sess.run(images)


    print("kkk")
    # Send request
    # See prediction_service.proto for gRPC request/response details.
    request = predict_pb2.PredictRequest()

    # Call GAN model to make prediction on the image
    request.model_spec.name = 'cifar10'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['image'].CopyFrom(  tf.contrib.util.make_tensor_proto(images_, shape=[128,48,48,3]))


    result = stub.Predict.future(request, 60.0)  # 60 secs timeout
    #print("result", result.result().outputs['class'])
    result_batch = result.result().outputs['class'].int64_val[0]
    print("result_batch", result_batch)


if __name__ == '__main__':
    tf.app.run()
