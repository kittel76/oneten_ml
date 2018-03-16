# coding=utf-8

import logging
from logging.handlers import TimedRotatingFileHandler


logger = logging.getLogger('mylogger')

fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')


# second (s)
# minute (m)
# hour (h)
# day (d)
# w0-w6 (weekday, 0=Monday)
# midnight


# fileHandler = logging.FileHandler('/data/log/similar_img_prd/process.log')
fileHandler = TimedRotatingFileHandler ('/data/log/create_rel_prd_img.log', when='m', interval=1)


streamHandler = logging.StreamHandler()

## handler에  포맷팅을 셋팅한다

fileHandler.setFormatter(fomatter)
streamHandler.setFormatter(fomatter)


logger.addHandler(fileHandler)
logger.addHandler(streamHandler)

logger.setLevel(logging.DEBUG)

logger.info("kkk")