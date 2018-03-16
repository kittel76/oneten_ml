# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os, time
import tensorflow as tf
from related_prd import dbio
import logging.config
import logging

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('related_prd')

os.putenv('NLS_LANG', '.UTF8')

def main(unused_argv):

    start = time.time()

    logger.info("start")

    conn = dbio.getConnection()
    curs = conn.cursor()
    sql= " \
     select to_char(p.ot_prd_no) \
     , replace( \
     c.lrg_cate_nm || \
     ' ' || c.midd_cate_nm || \
     ' ' || c.sml_cate_nm || \
     /* ' ' || nvl(oneten.FC_GET_PRD_ATTR_NM( p.ot_prd_no, 1), '') || */ \
     ' ' || nvl(oneten.FC_GET_PRD_ATTR_NM( p.ot_prd_no, 2), '') || \
     ' ' || nvl(oneten.FC_GET_PRD_ATTR_NM( p.ot_prd_no, 7),'') || \
     ' ' || nvl(oneten.FC_GET_PRD_disp_ATTR_NM( p.ot_prd_no, 3),'') || \
     ' ' || nvl(oneten.FC_GET_PRD_ATTR_NM( p.ot_prd_no, 4),'') || \
     ' ' || nvl(oneten.FC_GET_PRD_ATTR_NM( p.ot_prd_no, 5),'') || \
     ' ' || nvl(oneten.FC_GET_PRD_ATTR_NM( p.ot_prd_no, 6),'') || \
     ' ' || m.mall_nm || \
     ' ' || p.prd_nm  \
     , ',', ' ')   \
     as prd_txt \
    from oneten.prd_m p \
    inner join oneten.mall_m m on (p.mall_no = m.mall_no) \
    inner join oneten.cate_de_d c on (p.sml_cate_no = c.cate_no) \
    where p.prd_disp_flag = '02' and p.prd_use_flag = '02' \
    and SYSDATE BETWEEN  p.disp_start_dtm AND p.disp_end_dtm \
     \
    "

    curs.execute(sql)
    data = curs.fetchall()

    nd = np.array(data)

    logger.info("data size: %s", nd.__len__())

    with open("/data/www/oneten/dl_related_prd/prd_txt.csv", "w") as f:
        f.write("id,description\n")
        # np.savetxt(f, nd, fmt="%s", delimiter=",")
        for data in nd:
            f.write(data[0] + "," + "\"" + data[1] + "\""+ os.linesep)
        f.close()

    conn.close()

    end = time.time() - start
    logger.info("end")
    logger.info("elapsed time: %s", end)

if __name__ == '__main__':
  tf.app.run()