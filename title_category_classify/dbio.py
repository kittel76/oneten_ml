from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cx_Oracle
import os
import numpy as np
os.putenv('NLS_LANG', '.UTF8')


dbUrl = "db.stat.wishlink.info"
dbPort = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1522

def getConnection():
    return cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))




def getCateNms ():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))

    curs = conn.cursor()
    sql = "select  cate_no||'_'||replace(cate_nm, '/', '-') cate_nm, cate_no   from ml_cate_info order by idx   "
    curs.execute(sql)
    cate_nm = curs.fetchall()

    nd = np.array(cate_nm)

    return nd[:,[0]].flatten(), nd[:,[1]].flatten()
