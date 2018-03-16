from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cx_Oracle
import os
os.putenv('NLS_LANG', '.UTF8')


dbUrl = "db.main.wishlink.info"
dbPort = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1521


def getConnection():
    return cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

def getConnectionByOption(autocommit=0, threaded=False):
    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"), threaded=threaded)
    conn.autocommit = 1
    return conn

def truncate_related_prd_list_tmp():
    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs1 = conn.cursor()
    curs1.execute("truncate table batch_dl_rel_prd_d")
    conn.commit()

def truncate_related_prd_list():

    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_rel_prd_d")
    conn.commit()


def insert_related_prd_list_tmp(prd, item, conn):

    curs = conn.cursor()
    sql = "insert /*+append*/ all  "
    data_len = len(item)
    for i in range(0, data_len):
        tmp = " into batch_dl_rel_prd_d (ot_prd_no, sort_no,  rel_prd_no, reg_dtm, regr) values ( " + str(prd) + "," +  str(i+1) + "," + str(item[i]) + ", sysdate, 10000002 ) "
        sql += tmp
        # if i != (data_len - 1):
        #     sql += ","
    sql += " select * from dual "
    curs.execute(sql)


def insert_related_prd_list():


    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    sql = "insert into dl_rel_prd_d  \
            select * from batch_dl_rel_prd_d "
    curs.execute(sql)
    conn.commit()
