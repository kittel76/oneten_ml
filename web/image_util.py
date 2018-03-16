# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import tensorflow as tf
from datetime import datetime
import numpy as np
import cv2

def resize_and_crop(img_path, modified_path, size, crop_type='top'):
    """
    Resize and crop an image to fit the specified size.

    args:
        img_path: path for the image to resize.
        modified_path: path to store the modified image.
        size: `(width, height)` tuple.
        crop_type: can be 'top', 'middle' or 'bottom', depending on this
            value, the image will cropped getting the 'top/left', 'middle' or
            'bottom/right' of the image to fit the size.
    raises:
        Exception: if can not open the file in img_path of there is problems
            to save the image.
        ValueError: if an invalid `crop_type` is provided.
    """
    # If height is higher we resize vertically, if not we resize horizontally
    img = Image.open(img_path)
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    # ratio = 1
    #The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], round(size[0] * img.size[1] / img.size[0])),
                Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, round((img.size[1] - size[1]) / 2), img.size[0],
                   round((img.size[1] + size[1]) / 2))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize(( int( round(size[1] * img.size[0] / img.size[1]) ) , size[1]), Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (round((img.size[0] - size[0]) / 2), 0,
                   round((img.size[0] + size[0]) / 2), img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else :
        img = img.resize((size[0], size[1]),
                Image.ANTIALIAS)
        # If the scale is the same, we do not need to crop
    img.save(modified_path)



def resize(img_path, modified_path, size):

    img = Image.open(img_path)
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    ratio = 1
    #The image is scaled/cropped vertically or horizontally depending on the ratio

    img = img.resize((size[0], size[1]),
            Image.ANTIALIAS)
        # If the scale is the same, we do not need to crop
    img.save(modified_path)

def crop(img_path, modified_path, size, crop_type='middle'):

    img = Image.open(img_path)
    # Get current and desired ratio for the images

    #The image is scaled/cropped vertically or horizontally depending on the ratio

    # Crop in the top, middle or bottom
    if crop_type == 'top':
        box = (0, 0, size[0], img.size[1])
    elif crop_type == 'middle':
        box = (round((img.size[0] - size[0]) / 2), 0,
               round((img.size[0] + size[0]) / 2), img.size[1])
    elif crop_type == 'bottom':
        box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
    else:
        raise ValueError('ERROR: invalid value for crop_type')
    img = img.crop(box)
    img.save(modified_path)

def gifToJpeg(infile, newfile):
    try:
        im = Image.open(infile)
    except IOError:
        print ("Cant load", infile)
    i = 0
    mypalette = im.getpalette()

    try:
        im.putpalette(mypalette)
        new_im = Image.new("RGB", im.size)
        new_im.paste(im)
        new_im.save(newfile)

    except EOFError:
        pass # end of sequence



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



def get_process_image(filename, width, height):

  image = decode_jpeg(filename)
  # 텐서로 변환
  reshaped_image = tf.cast(image, tf.float32)
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)

  float_image = tf.image.per_image_standardization(resized_image)
  return float_image

def decode_jpeg(image_file):
  from PIL import Image
  im = Image.open(image_file)
  data = np.array(im)
  return data


def get_process_image_with_sess(sess, filename, width, height):

  # # image = np.asarray(Image.open(filename))
  # with tf.gfile.FastGFile(filename, 'rb') as f:
  #   image_data = f.read()
  #
  # decoded_image = tf.image.decode_jpeg(image_data, channels=3)
  # image = sess.run(decoded_image)

  image = decode_jpeg(filename)
  # 텐서로 변환
  reshaped_image = tf.cast(image, tf.float32)
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)

  float_image = tf.image.per_image_standardization(resized_image)

  return float_image


def get_process_image_with_tensor(image, width, height):


  reshaped_image = tf.cast(image, tf.float32)
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         height, width)

  float_image = tf.image.per_image_standardization(resized_image)

  return float_image


def get_process_image_real(filename):

  image = np.float32(decode_jpeg(filename))
  mean = np.mean(image)
  stddev = np.std( image)

  num_pixels = np.prod(image.shape)
  min_stddev = np.float32( 1 / np.sqrt(num_pixels))
  adjusted_stddev = np.maximum (stddev,  min_stddev)

  float_image =  (image - mean) / adjusted_stddev

  return float_image


def get_img_tensor(images):
    print("step2-0", datetime.now())

    sess = tf.Session()
    images_ = sess.run(images)

    return images_