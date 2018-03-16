from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from batch import dbio_move
import logging.config
logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('batch')


def main():

    dbio_move.truncate_user_item_rank_batch()
    # dbio_move.move_to_real()
    dbio_move.move_to_real_by_dblink()


if __name__ == '__main__':
    total_start_time = time.time();
    main()
    total_end_time = time.time();

    logger.info("total elapsed time:%s",total_end_time - total_start_time )
    logger.info("Done!")