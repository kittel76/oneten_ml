from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os, time
import logging.config
from title_category_classify import dbio
from title_category_classify import common
from util import dir_util
os.putenv('NLS_LANG', '.UTF8')


logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('title_category_classify')


def main():

    conn = dbio.getConnection()

    curs = conn.cursor()
    sql= " select \
     p.prd_nm \
     ,c.idx \
    from stat_ot.prd_m p \
    inner join ml_cate_info c on (p.sml_cate_no = c.cate_no) \
    where p.prd_use_flag = '02' AND p.prd_disp_flag not in ('01' ) \
     and mod(p.ot_prd_no,10) > 0  \
     \
    "

    curs.execute(sql)
    data = curs.fetchall()

    nd = np.array(data)


    curs = conn.cursor()
    sql= " select \
     p.prd_nm \
     ,c.idx \
    from stat_ot.prd_m p \
    inner join ml_cate_info c on (p.sml_cate_no = c.cate_no) \
    where p.prd_use_flag = '02' AND p.prd_disp_flag not in ('01' )  \
     and mod(p.ot_prd_no,10) = 0  \
     \
    "

    curs.execute(sql)
    data = curs.fetchall()
    nd_test = np.array(data)

    # print(nd)

    dir_util.mk_dir_recursive(common.DATA_DIR)
    with open(common.DATA_DIR + "/data.csv", "w") as f:
        f.write("prd_nm,idx\n")
        # np.savetxt(f, nd, fmt="%s", delimiter=",")
        for data in nd:
            f.write(data[0] + "###" +  data[1] +  os.linesep)
        f.close()

    logger.info("complete write " + common.DATA_DIR + "/data.csv")

    with open(common.DATA_DIR + "/data_test.csv", "w") as f:
        f.write("prd_nm,idx\n")
        # np.savetxt(f, nd, fmt="%s", delimiter=",")
        for data in nd_test:
            f.write(data[0] + "###" +  data[1] +  os.linesep)
        f.close()




    cate_nms, cate_nos = dbio.getCateNms()
    with open(common.DATA_DIR + "/labels.txt", "w") as f:
        for (i, cate_no) in zip(cate_nms, cate_nos):
            f.write(i + os.linesep)


if __name__ == '__main__':

    logger.info("start")
    start = time.time();
    main()

    end = time.time() - start

    logger.info("end")
    logger.info("elased time:%s", end)

