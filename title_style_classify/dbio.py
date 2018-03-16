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
    return cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))




def getStyleNms ():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    sql = "select attr_no||'_'||attr_nm, attr_no from ONETEN.ATTR_M where attr_grp_no = 2 order by sort_no  "
    curs.execute(sql)
    cate_nm = curs.fetchall()

    nd = np.array(cate_nm)
    return nd[:, [0]].flatten(), nd[:, [1]].flatten()


