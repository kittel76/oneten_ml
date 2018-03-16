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

os.putenv('NLS_LANG', '.UTF8')
dbUrl = "db.stat.wishlink.info"
dbPort = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1522


logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('cf_with_rbm')


def dbToFile(targetDate):
    logger.info("targetDate:%s", targetDate)

    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))

    curs = conn.cursor()
    sql = \
        "select t.idx_cust_no, t.idx_prd_no, round(t.rating * t.apply_rt , 2) as rating  from ( \
        select u.idx_cust_no, p.idx_prd_no,  \
        	 case  \
                when a.ng_cnt > 0 then 0   \
                when a.detail_click_cnt > 0 then 15   \
                when a.click_cnt > 0 and a.like_cnt > 0 then 13 \
                when a.click_cnt > 0 or a.ep_click_cnt > 0  then 12   \
                when a.like_cnt > 0 then 10   \
        	    else 1 end rating   \
        	    , 1.0 as  apply_rt  \
        	 from dl_cust_prd_info_target a    \
              inner join dl_cust_map u on (a.cust_no = u.cust_no)   \
              inner join dl_prd_map p on (a.ot_prd_no = p.ot_prd_no)  \
        ) t    \
        "

    curs.execute(sql)
    data = curs.fetchall()

    nd = np.array(data)

    target_dir = "/data/www/oneten/data_cf_with_rbm/data_dir/" + targetDate
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    with open(target_dir + "/rating.csv", "wb") as f:
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