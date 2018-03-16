from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cx_Oracle
import os
os.putenv('NLS_LANG', '.UTF8')

dbUrl = "db.stat.wishlink.info"
dbPort = 1521

dbUrlForReal = "db.main.wishlink.info"
dbPortForReal = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1522
    dbUrlForReal = "hostway.gate.wishlink.info"
    dbPortForReal = 1521


def getConnection():
    return cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))


def getConnectionByOption(autocommit=0, threaded=False):
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"), threaded=threaded)
    conn.autocommit = 1
    return conn


def getConnectionOneten():
    return cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrlForReal, dbPortForReal, "oraOT1"))


def truncate_dl_cust_prd_info_target():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_prd_info_target")
    conn.commit()
    conn.close()


def truncate_dl_cust_target():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_target")
    conn.commit()
    conn.close()


def truncate_user_item_rank_tmp():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_prd_rank_tmp")
    conn.commit()
    conn.close()


def truncate_user_item_rank_tmp2():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_prd_rank_tmp2")
    conn.commit()
    conn.close()

def truncate_user_item_rank_thread():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_prd_rank_thread")
    conn.close()

def truncate_user_item_rank():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_prd_rank")
    conn.commit()
    conn.close()

def truncate_user_item_rank_a():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_prd_rank_a")
    conn.commit()
    conn.close()


def insert_user_item_rank_tmp(user, item, conn):
    curs = conn.cursor()
    sql = "insert  /*+append*/ all  "

    data_len = len(item)
    for i in range(0, data_len):
        tmp = " into dl_cust_prd_rank_tmp values ( " + str(user) + "," + str(item[i]) + "," + str(i+1) + " ) "
        sql += tmp
        # if i != (data_len - 1):
        #     sql += ","
    sql += " select * from dual "
    curs.execute(sql)
        # print("idx:", user)
    conn.commit()

def insert_user_item_rank_tmp_nocommit(user, item, conn):
    curs = conn.cursor()
    sql = "insert  /*+append*/ all  "

    data_len = len(item)
    for i in range(0, data_len):
        tmp = " into dl_cust_prd_rank_tmp values ( " + str(user) + "," + str(item[i]) + "," + str(i+1) + " ) "
        sql += tmp
        # if i != (data_len - 1):
        #     sql += ","
    sql += " select * from dual "
    curs.execute(sql)

def insert_user_item_rank_tmp2(user, item, conn):
    curs = conn.cursor()
    sql = "insert  /*+append*/ all  "

    data_len = len(item)
    for i in range(0, data_len):
        tmp = " into dl_cust_prd_rank_tmp2 values ( " + str(user) + "," + str(item[i]) + "," + str(i+1) + " ) "
        sql += tmp
        # if i != (data_len - 1):
        #     sql += ","
    sql += " select * from dual "

    print("connInsert.authcommit", conn.autocommit)
    curs.execute(sql)


def insert_user_item_rank_thread(user, item, conn):
    curs = conn.cursor()
    sql = "insert all  "

    data_len = len(item)
    for i in range(0, data_len):
        tmp = " into DL_CUST_PRD_RANK_thread values ( " + str(user) + "," + str(item[i]) + "," + str(i+1) + " ) "
        sql += tmp
        # if i != (data_len - 1):
        #     sql += ","
    sql += " select * from dual "
    curs.execute(sql)


def insert_uesr_item_rank(limit_size=40):
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs = conn.cursor()
    sql = "insert into DL_CUST_PRD_RANK (cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid) \
            select b.cust_no, c.ot_prd_no, a.sort_no, sysdate , '2', " + str(limit_size)  + " + 1 - sort_no, 'oneten' sid from DL_CUST_PRD_RANK_TMP a  \
            inner join dl_cust_map b on (a.idx_cust_no = b.idx_cust_no)  \
            inner join dl_prd_map c on (a.idx_prd_no = c.idx_prd_no)  \
              "

    curs.execute(sql)
    conn.commit()
    conn.close()


def insert_real_user_item_rank():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs = conn.cursor()
    sql = "insert into DL_CUST_PRD_RANK (cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid) \
            select cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid from DL_CUST_PRD_RANK_A \
          "
    curs.execute(sql)


    conn.commit()
    conn.close()


def insert_user_item_rank_a(targetDate, limit_size=40):
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs = conn.cursor()
    sql = "insert /*+append*/ into DL_CUST_PRD_RANK_A (cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid) \
            select a.cust_no, a.ot_prd_no, a.sort_no, sysdate, '2_a',  " + str(limit_size)  + " + 1 - a.sort_no, a.sid  from ( \
                select  /*+ index(c idx01_dl_prd_map) */ b.cust_no, c.ot_prd_no, ROW_NUMBER() OVER (partition by a.idx_cust_no order by a.sort_no) sort_no, d.sid from DL_CUST_PRD_RANK_TMP a  \
                inner join dl_cust_map b on (a.idx_cust_no = b.idx_cust_no)  \
                inner join dl_prd_map c on (a.idx_prd_no = c.idx_prd_no)  \
                inner join stat_ot.cust_m d on (b.cust_no = d.cust_no) \
                inner join stat_ot.prd_m p on (p.ot_prd_no = c.ot_prd_no and p.prd_use_flag = '02' and prd_disp_flag = '02'  and sysdate between p.disp_start_dtm and p.disp_end_dtm ) \
            ) a \
              "

    curs.execute(sql)
    conn.commit()
    conn.close()


def getMaxUserNum (targetDate):
    conn = getConnection()

    curs = conn.cursor()
    sql = "select max(idx_cust_no) max_cust_no from dl_cust_map where target_date = '" + targetDate + "'"

    max_user_id = 0
    curs.execute(sql)
    (max_user_id,) = curs.fetchone()

    # conn.close();
    return max_user_id


def getMaxItemNum (targetDate):
    conn = getConnection()

    curs = conn.cursor()
    sql = "select max(idx_prd_no) max_prd_no from dl_prd_map where  target_date = '" + targetDate + "'"

    max_prd_no = 0
    curs.execute(sql)
    (max_prd_no,) = curs.fetchone()

    # conn.close()
    return max_prd_no


def is_target_user_thread(idx_cust_no):
    conn = getConnection()
    curs = conn.cursor()
    sql = " select count(*) cnt, sum(rec_cnt) rec_cnt from  dl_cust_map a inner join dl_cust_target b on (a.cust_no = b.cust_no) where a.idx_cust_no = :1  group by a.idx_cust_no "
    result = curs.execute(sql, (int(idx_cust_no),))
    cnt = 0
    rec_cnt = 0
    #    (cnt,rec_cnt) = result.fetchone()
    while True:
        row = result.fetchone()
        if row == None:
            break;
        (cnt, rec_cnt) = row

    conn.close()
    if int(cnt) > 0:
        return (True, int(rec_cnt))
    else:
        return (False, int(rec_cnt))



def is_target_user(conn, idx_cust_no, target_date):


    curs = conn.cursor()
    sql = " select count(*) cnt, sum(rec_cnt) rec_cnt from  dl_cust_map a inner join dl_cust_target b on (a.cust_no = b.cust_no) where a.idx_cust_no = :1  group by a.idx_cust_no "
    result = curs.execute(sql, (int(idx_cust_no),))
    cnt = 0
    rec_cnt = 0
    #    (cnt,rec_cnt) = result.fetchone()
    while True:
        row = result.fetchone()
        if row == None:
            break;
        (cnt, rec_cnt) = row

    if int(cnt) > 0:
        return (True, int(rec_cnt))
    else:
        return (False, int(rec_cnt))






def getTargetDate ():
    conn = getConnection()
    curs = conn.cursor()
    sql = "select target_date from dl_cust_map where rownum = 1"

    curs.execute(sql)
    (target_date,) = curs.fetchone()

    # conn.close();
    return target_date