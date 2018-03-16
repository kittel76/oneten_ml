from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os, time
import logging.config
from search_keyword import dbio
from util import dir_util
os.putenv('NLS_LANG', '.UTF8')


logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('title_category_classify')

DATA_DIR = "/data/www/oneten/data/search_data"

def main():

    conn = dbio.getConnection()

    curs = conn.cursor()
    sql= " select \
     p.prd_nm \
    from oneten.prd_m p \
    inner join stat.ml_cate_info c on (p.sml_cate_no = c.cate_no) \
    where p.prd_disp_flag = '02' and p.use_yn = 'Y' and p.exm_flag = '03' and p.soldout_yn = 'N' \
     \
    "

    curs.execute(sql)
    data = curs.fetchall()

    nd = np.array(data)

    # with open("." + "/data.txt", "w") as f:
    #     np.savetxt(f, nd, fmt="%s", delimiter=",")
    with open(DATA_DIR + "/data.txt", "w") as f:
        # np.savetxt(f, nd, fmt="%s", delimiter=",")
        idx = 0
        for data in nd:
            if (idx ==0):
                f.write(data[0])
            else:
                f.write(os.linesep + data[0].lower())
            idx = idx + 1
        f.close()


if __name__ == '__main__':

    logger.info("start")
    start = time.time();
    main()

    end = time.time() - start

    logger.info("end")
    logger.info("elased time:%s", end)

