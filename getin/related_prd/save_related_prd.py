from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from related_prd.engines import content_engine
import logging.config
import logging

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('related_prd')

def main(unused_argv):

    start = time.time()
    logger.info("start")

    content_engine.train('/data/www/oneten/dl_related_prd/prd_txt.csv')

    end = time.time() - start
    logger.info("end")
    logger.info("elasped time: %s", end)


if __name__ == '__main__':
  tf.app.run()