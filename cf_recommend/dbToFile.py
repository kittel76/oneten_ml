from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cx_Oracle
import numpy as np
import os
import argparse
from datetime import date
import time


parser = argparse.ArgumentParser()
parser.add_argument("--targetDate", help="targetDate")
args = parser.parse_args()
if args.targetDate == None:
    args.targetDate = date.fromtimestamp(time.time() + 60 * 60 * 24).strftime('%Y%m%d')


print ("args.targetDate", args.targetDate)
os.putenv('NLS_LANG', '.UTF8')

dbUrl = "db.main.wishlink.info"
dbPort = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1521

import logging.config
import logging
logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('cf_recommend')

def convert(x):
  try:
    return x.astype(int)
  except:
    return x




conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

curs = conn.cursor()
sql = \
    "select t.idx_cust_no, t.idx_prd_no, round(t.rating * t.apply_rt , 2) as rating  from ( \
    select u.idx_cust_no, p.idx_prd_no,  \
    	 case  \
            when a.ng_cnt > 0 then 0   \
            when a.detail_click_cnt > 0 then 15   \
            when a.like_cnt > 0 then 12   \
            when a.click_cnt > 0 then 10   \
            when a.ep_click_cnt > 0 then 10   \
    	    else 1 end rating   \
    	, case when   (sysdate - pm.reg_dtm <= 7 ) then 1.0 else 1.0 end  apply_rt  \
    	 from dl_cust_prd_info_target a    \
           inner join dl_cust_map u on (a.cust_no = u.cust_no)   \
          inner join dl_prd_map p on (a.ot_prd_no = p.ot_prd_no)  \
    	  inner join oneten.prd_m pm on (a.ot_prd_no = pm.ot_prd_no)   \
    ) t    \
    "


curs.execute(sql)
data = curs.fetchall()

nd = np.array(data)

target_dir = "/data/www/oneten/dl_cf_recommend/data_dir/" + args.targetDate
if not os.path.isdir(target_dir):
    os.mkdir(target_dir)

with open(target_dir + "/click.csv", "wb") as f:
    np.savetxt(f, nd, fmt='%i', delimiter=",")
    f.close()

with open(target_dir + "/item.csv", "wb") as f:
    np.savetxt(f, np.unique(nd[:,1]), fmt='%i', delimiter=",")
    f.close()

with open(target_dir + "/user.csv", "wb") as f:
    np.savetxt(f, np.unique(nd[:,0]), fmt='%i', delimiter=",")
    f.close()
