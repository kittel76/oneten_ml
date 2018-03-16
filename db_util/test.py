# coding=utf-8
import time

from db_util import log_util

now = time.localtime()
dates = str(now.tm_year) + str(now.tm_mon).rjust(2,'0') + str(now.tm_mday).rjust(2,'0')

# 가나다
print(dates)


print("테스트"
      "")
