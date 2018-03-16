from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os


tf.app.flags.DEFINE_string('IMAGE_SIZE', 48, """Process images of this size.""")
tf.app.flags.DEFINE_string('NUM_CLASSES', 15, """NUM_CLASSES""")



tf.app.flags.DEFINE_string('train_directory', '/data/www/oneten/dl_img_detail_process2/data_dir/','Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/data/www/oneten/dl_img_detail_process2/data_dir/',  'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/data/www/oneten/dl_img_detail_process2/data_dir',  'Output data directory')
tf.app.flags.DEFINE_string('labels_file', '/data/www/oneten/dl_img_detail_process2/data_dir/labels.txt', 'Labels file')


tf.app.flags.DEFINE_string('train_directory_for_test', '/data/www/oneten/dl_img_detail_process2/test_data_dir/','Training data directory')
tf.app.flags.DEFINE_string('validation_directory_for_test', '/data/www/oneten/dl_img_detail_process2/test_data_dir/',  'Validation data directory')
tf.app.flags.DEFINE_string('output_directory_for_test', '/data/www/oneten/dl_img_detail_process2/test_data_dir',  'Output data directory')
tf.app.flags.DEFINE_string('labels_file_for_test', '/data/www/oneten/dl_img_detail_process2/test_data_dir/labels.txt', 'Labels file')



tf.app.flags.DEFINE_integer('train_shards', 3, 'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 3, 'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 3,  'Number of threads to preprocess the images.')

# cifar10_train.py
tf.app.flags.DEFINE_string('train_dir', '/data/www/oneten/dl_img_detail_process2/train_dir/', 'Directory where to write event logs  and checkpoint.')
tf.app.flags.DEFINE_integer('max_steps', 100000, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,  """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100, """How often to log results to the console.""")


tf.app.flags.DEFINE_string('image_size', '64x64', 'training image download image size')


tf.app.flags.DEFINE_integer('src_image_width', 64, 'training image download image size')
tf.app.flags.DEFINE_integer('src_image_height', 64, 'training image download image size')


# tf.app.flags.DEFINE_string('eval_data', 'test',  'Either  or train_eval')
tf.app.flags.DEFINE_string('checkpoint_dir', '/data/www/oneten/dl_img_detail_process2/train_dir/', 'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 5, 'How often to run the eval.')
tf.app.flags.DEFINE_integer('num_examples', 10000,  'Number of examples to run.')
tf.app.flags.DEFINE_boolean('run_once', True,  'Whether to run eval only once.')


tf.app.flags.DEFINE_integer('batch_size', 128,  """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/data/www/oneten/dl_img_detail_process2/',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")


dbUrl = 'db.main.wishlink.info'
dbPort = 1521
if os.getenv("pythonAppType", "") == "local":
    dbUrl = 'hostway.gate.wishlink.info'
    dbPort = 1521
tf.app.flags.DEFINE_string('dbUrl', dbUrl, 'database address')
tf.app.flags.DEFINE_integer('dbPort', dbPort, 'database port number')



from os.path import join as join_paths

def mk_dir_recursive(dir_path):

    if os.path.isdir(dir_path):
        return
    h, t = os.path.split(dir_path)  # head/tail
    if not os.path.isdir(h):
        mk_dir_recursive(h)

    new_path = join_paths(h, t)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
