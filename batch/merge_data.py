from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from batch import ops
from batch import dbio


np.random.seed(13575)

DIM = ops.DIM
DEVICE = "/cpu:0"


import logging.config
logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('batch')

def svd():

    dbio.truncate_user_item_rank()
    dbio.insert_real_user_item_rank()


if __name__ == '__main__':
    total_start_time = time.time();
    svd()
    total_end_time = time.time();
    logger.info("total elapsed time:%s",total_end_time - total_start_time )
    logger.info("Done!")