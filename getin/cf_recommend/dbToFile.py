from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import date
import os
import numpy as np
import cx_Oracle
import logging.config
import logging
from getin.cf_recommend import common

os.putenv('NLS_LANG', '.UTF8')
dbUrl = "db.stat.wishlink.info"
dbPort = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1522

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('cf_recommend')


def dbToFile(targetDate):
    logger.info("targetDate:%s", targetDate)

    conn = cx_Oracle.connect("getin", "getin2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))

    curs = conn.cursor()
    sql = \
        "select t.idx_gi_cust_no, t.idx_prd_no, t.rating  from ( \
        select u.idx_gi_cust_no, p.idx_prd_no,  \
        	 case when a.prd_dtl_click_cnt > 1 then 20 else 10 end rating \
        from dl_cust_prd_target_d a    \
               inner join dl_cust_map_d u on (a.gi_cust_no = u.gi_cust_no)   \
              inner join dl_prd_map_d p on (a.prd_no = p.prd_no and a.mall_no = p.mall_no)  \
        ) t    \
        "

    logger.info("sql:%s", sql)
    curs.execute(sql)
    data = curs.fetchall()

    nd = np.array(data)

    target_dir = common.DATA_HOME + "/data_dir/" + targetDate
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    with open(target_dir + "/click.csv", "wb") as f:
        np.savetxt(f, nd, fmt='%i', delimiter=",")
        f.close()

    with open(target_dir + "/item.csv", "wb") as f:
        np.savetxt(f, np.unique(nd[:, 1]), fmt='%i', delimiter=",")
        f.close()

    with open(target_dir + "/user.csv", "wb") as f:
        np.savetxt(f, np.unique(nd[:, 0]), fmt='%i', delimiter=",")
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--targetDate", help="targetDate")
    args = parser.parse_args()
    if args.targetDate == None:
        logger.info("args.targetDate == None")
        args.targetDate = date.fromtimestamp(time.time() + 60 * 60 * 24).strftime('%Y%m%d')

    logger.info ("args.targetDate:%s", args.targetDate)

    dbToFile(args.targetDate)