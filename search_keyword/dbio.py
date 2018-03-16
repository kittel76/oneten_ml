from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cx_Oracle
import os
import numpy as np
os.putenv('NLS_LANG', '.UTF8')


dbUrl = "db.main.wishlink.info"
dbPort = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1521


def getConnection():
    return cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))





def truncate_search_sggt_prdnm_freq_d():
    conn = getConnection()
    curs1 = conn.cursor()
    curs1.execute("truncate table search_sggt_prdnm_freq_d")
    conn.commit()
    conn.close()


def insert_user_item_rank_batch(item, conn):
    curs = conn.cursor()
    sql = "insert all  "
    data_len = len(item)
    for i in range(0, data_len):
        tmp = " into SEARCH_SGGT_PRDNM_FREQ_D values \
            ( "\
              + "'" + item[i][0]  + "', sysdate, '" + str(item[i][1])  + "'" \
              " ) "
        sql += tmp
        # if i != (data_len - 1):
        #     sql += ","
    sql += " select * from dual "
    curs.execute(sql)
        # print("idx:", user)
    conn.commit()