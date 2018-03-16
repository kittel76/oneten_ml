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
    return cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))


def truncate_uesr_item_rank_tmp():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_prd_rank_b_tmp")
    conn.commit()
    conn.close()


def truncate_related_prd_rank_tmp():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table DL_CFM_RELATED_PRD_RANK_TMP")
    conn.commit()
    conn.close()


def truncate_uesr_item_rank_b():
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_prd_rank_b")
    conn.commit()
    conn.close()


def insert_uesr_item_rank_tmp(user, item, conn):
    curs = conn.cursor()
    sql = "insert all  "

    data_len = len(item)
    for i in range(0, data_len):
        tmp = " into dl_cust_prd_rank_b_tmp values ( " + str(user) + "," + str(item[i]) + "," + str(i+1) + " ) "
        sql += tmp
        # if i != (data_len - 1):
        #     sql += ","
    sql += " select * from dual "
    # print(sql)
    curs.execute(sql)
        # print("idx:", user)
    conn.commit()


def insert_related_prd_rank_tmp(otPrdNo, item, conn):
    curs = conn.cursor()
    sql = "insert all  "

    data_len = len(item)
    for i in range(0, data_len):
        tmp = " into DL_CFM_RELATED_PRD_RANK_TMP values ( " + str(otPrdNo) + "," + str(item[i]) + "," + str(i+1) + " ) "
        sql += tmp
        # if i != (data_len - 1):
        #     sql += ","
    sql += " select * from dual "
    # print(sql)
    curs.execute(sql)
        # print("idx:", user)
    conn.commit()


def insert_uesr_item_rank(limit_size=40):
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs = conn.cursor()
    sql = "insert into DL_CUST_PRD_RANK_B (cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid) \
            select b.cust_no, c.ot_prd_no, a.sort_no, sysdate , '2', " + str(limit_size)  + " + 1 - sort_no, d.sid from DL_CUST_PRD_RANK_B_TMP a  \
            inner join dl_cust_map b on (a.idx_cust_no = b.idx_cust_no)  \
            inner join dl_prd_map c on (a.idx_prd_no = c.idx_prd_no)  \
            inner join oneten.cust_m d on (b.cust_no = d.cust_no) \
              "


    curs.execute(sql)
    conn.commit()
    conn.close()

def insert_uesr_item_rank_b(limit_size=40):
    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs = conn.cursor()
    sql = "insert into DL_CUST_PRD_RANK_B (cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid) \
            select b.cust_no, c.ot_prd_no, a.sort_no, sysdate , '2_b', " + str(limit_size)  + " + 1 - sort_no, 'oneten' sid from DL_CUST_PRD_RANK_B_TMP a  \
            inner join dl_cust_map b on (a.idx_cust_no = b.idx_cust_no)  \
            inner join dl_prd_map c on (a.idx_prd_no = c.idx_prd_no)  \
              "


    curs.execute(sql)
    conn.commit()
    conn.close()

#


def getMaxUserNum (targetDate):
    conn = getConnection()

    curs = conn.cursor()
    sql = "select max(idx_cust_no) max_cust_no from dl_cust_map where target_date = '" + targetDate + "'"

    max_user_id = 0
    curs.execute(sql)
    (max_user_id,) = curs.fetchone()

    conn.close();
    return max_user_id


def getMaxItemNum (targetDate):
    conn = getConnection()

    curs = conn.cursor()
    sql = "select max(idx_prd_no) max_prd_no from dl_prd_map where  target_date = '" + targetDate + "'"

    max_prd_no = 0
    curs.execute(sql)
    (max_prd_no,) = curs.fetchone()

    conn.close()
    return max_prd_no



def getIdxCustNo (cust_no):
    conn = getConnection()

    curs = conn.cursor()
    sql = "select idx_cust_no  from dl_cust_map where cust_no = :1 "
    idx_cust_no = 0
    curs.execute(sql, (cust_no,))

    (idx_cust_no,) = curs.fetchone()

    conn.close()
    return idx_cust_no



def is_target_user(conn, idx_cust_no):
    try:

        curs = conn.cursor()
        sql = " select count(*) cnt, sum(rec_cnt) rec_cnt from  dl_cust_map a inner join dl_cust_target b on (a.cust_no = b.cust_no) where a.idx_cust_no = :1  group by a.idx_cust_no "
        result = curs.execute(sql, (int(idx_cust_no),))
        cnt = 0
        rec_cnt =0
    #    (cnt,rec_cnt) = result.fetchone()
        while True:
            row = result.fetchone()
            if row ==None:
                break;
            (cnt, rec_cnt) = row

        if int(cnt) > 0:
            return (True, int(rec_cnt))
        else:
            return (False, int(rec_cnt))
    except Exception as ex:
        return (False, 0)


def getRecPrd (ot_prd_no):
    conn = getConnection()

    curs = conn.cursor()
    sql = "select idx_cust_no  from dl_cust_map where cust_no = :1 "
    idx_cust_no = 0
    curs.execute(sql, (cust_no,))

    (idx_cust_no,) = curs.fetchone()

    conn.close()
    return idx_cust_no