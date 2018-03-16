

import logging.config
import logging


logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('similar_img_prd')


logger.info("kkkk")