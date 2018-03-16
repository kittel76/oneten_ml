from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cx_Oracle
import os
import logging.config
os.putenv('NLS_LANG', '.UTF8')


logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('batch')

dbUrl = "db.stat.wishlink.info"
dbPort = 1521

dbUrlForReal = "db.main.wishlink.info"
dbPortForReal = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1521
    dbUrlForReal = "hostway.gate.wishlink.info"
    dbPortForReal = 1521


def getConnection():
    return cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))


def getConnectionOneten():
    return cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrlForReal, dbPortForReal, "oraOT1"))



def truncate_user_item_rank_batch():
    conn = getConnectionOneten()
    curs1 = conn.cursor()
    curs1.execute("truncate table OT_STAT.DL_CUST_PRD_RANK_BATCH")
    conn.commit()
    conn.close()


def move_to_real():
    conn = getConnection()

    dataList = []

    curs1 = conn.cursor()
    curs1.execute("select * from dl_cust_prd_rank where rownum <= 100000000 ")

    data = curs1.fetchall()
    conn.close()
    print("len(data)", len(data))

    dataLen = len(data)
    unitSize = 50

    dataSize = int(dataLen/unitSize)
    if dataLen%unitSize >0:
        dataSize = dataSize + 1
    logger.info("dataSize:%s", dataSize)

    conn_for_oneten = getConnectionOneten()
    for i in range(0, dataSize):
        startIdx = unitSize*i
        endIdx = unitSize*(i+1)
        if endIdx > (dataLen ):
            endIdx = dataLen
        if i%100 == 0:
            logger.info("startIdx:%s, endIdx:%s", startIdx, endIdx)
        # print(data[startIdx:endIdx])
        insert_user_item_rank_batch(data[startIdx:endIdx], conn_for_oneten)

    # conn_for_oneten.close()

def move_to_real_by_dblink():
    conn = getConnection()
    curs1 = conn.cursor()
    curs1.execute("insert into ot_stat.dl_cust_prd_rank_batch@ot.oneten (cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid) select cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid from dl_cust_prd_rank ")
    conn.commit()
    conn.close()

def insert_user_item_rank_batch(item, conn):
    conn = getConnectionOneten()
    curs = conn.cursor()
    sql = "insert all  "

    data_len = len(item)
    for i in range(0, data_len):
        tmp = " into OT_STAT.dl_cust_prd_rank_batch values \
            ( "\
              + str(item[i][0]) + "," +  str(item[i][1])  + "," +  str(item[i][2])  + ",to_date('" +  str(item[i][3])  + "', 'yyyy-mm-dd hh24:mi:ss'),'" +  str(item[i][4])  + "'," +  str(item[i][5])  + ",'" +  str(item[i][6])  + "'" \
              " ) "
        sql += tmp
        # if i != (data_len - 1):
        #     sql += ","
    sql += " select * from dual "

    curs.execute(sql)
        # print("idx:", user)
    conn.commit()



def replace_data():
    conn = getConnectionOneten()
    curs1 = conn.cursor()

    curs1.execute("truncate table OT_STAT.DL_CUST_PRD_RANK")
    curs1.execute("insert into OT_STAT.DL_CUST_PRD_RANK select * from OT_STAT.DL_CUST_PRD_RANK_batch")


    conn.commit()
    conn.close()

def replace_data_target_cust():
    conn = getConnectionOneten()
    curs1 = conn.cursor()

    curs1.execute("truncate table ot_stat.dl_cust_target")
    curs1.execute("insert into ot_stat.dl_cust_target select * from dl_cust_target@st.stat")

    conn.commit()
    conn.close()



def getDlCustPrdRankBatchCnt ():
    conn = getConnectionOneten()

    curs = conn.cursor()
    sql = "select count(*) from OT_STAT.DL_CUST_PRD_RANK_batch"

    cnt = 0
    curs.execute(sql)
    (cnt,) = curs.fetchone()

    # conn.close();
    return cnt