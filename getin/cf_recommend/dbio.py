from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cx_Oracle
import os
os.putenv('NLS_LANG', '.UTF8')

dbUrl = "db.stat.wishlink.info"
dbPort = 1521

# dbUrlForReal = "db.main.wishlink.info"
# dbPortForReal = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1522


def getConnection():
    return cx_Oracle.connect("getin", "getin2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))


# def getConnectionOneten():
#     return cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrlForReal, dbPortForReal, "oraOT1"))


def truncate_dl_cust_prd_info_target():
    conn = cx_Oracle.connect("getin", "getin2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_prd_info_target")
    conn.commit()
    conn.close()


def truncate_dl_cust_target():
    conn = cx_Oracle.connect("getin", "getin2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_target")
    conn.commit()
    conn.close()


def truncate_user_item_rank_tmp():
    conn = cx_Oracle.connect("getin", "getin2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table bt_dl_cust_prd_rank_d")
    conn.commit()
    conn.close()

def truncate_uesr_item_rank():
    conn = cx_Oracle.connect("getin", "getin2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs1 = conn.cursor()
    curs1.execute("truncate table dl_cust_prd_rank_d")
    conn.commit()
    conn.close()


def insert_user_item_rank_tmp(user, item, conn, dates):
    curs = conn.cursor()
    sql = "insert all  "

    data_len = len(item)
    for i in range(0, data_len):
        tmp = " into bt_dl_cust_prd_rank_d values ( " + str(user) + "," + str(item[i]) + "," + str(i+1) + "," + "'" + dates + "'" + ") "
        sql += tmp
        # if i != (data_len - 1):
        #     sql += ","
    sql += " select * from dual "
    curs.execute(sql)
        # print("idx:", user)
    conn.commit()


def insert_uesr_item_rank(limit_size=40):
    conn = cx_Oracle.connect("getin", "getin2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs = conn.cursor()
    sql = "insert into DL_CUST_PRD_RANK (cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid) \
            select b.cust_no, c.ot_prd_no, a.sort_no, sysdate , '2', " + str(limit_size)  + " + 1 - sort_no, 'oneten' sid from DL_CUST_PRD_RANK_TMP a  \
            inner join dl_cust_map b on (a.idx_gi_cust_no = b.idx_gi_cust_no)  \
            inner join dl_prd_map c on (a.idx_prd_no = c.idx_prd_no)  \
              "

    curs.execute(sql)
    conn.commit()
    conn.close()


def insert_real_user_item_rank():
    conn = cx_Oracle.connect("getin", "getin2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs = conn.cursor()
    sql = "insert into DL_CUST_PRD_RANK (cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid) \
            select cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid from DL_CUST_PRD_RANK_A \
            where mod(cust_no, 10) > 0 \
          "
    curs.execute(sql)

    curs1 = conn.cursor()
    sql = " insert into DL_CUST_PRD_RANK (cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid) \
            select cust_no, ot_prd_no, sort_no, regdt, rec_ver, score, sid from DL_CUST_PRD_RANK_A \
            where mod(cust_no, 10) = 0 \
          "
    curs1.execute(sql)

    conn.commit()
    conn.close()


def insert_user_item_rank(targetDate, limit_size=40):
    conn = cx_Oracle.connect("getin", "getin2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs = conn.cursor()
    sql = "insert into DL_CUST_PRD_RANK_d (gi_cust_no, mall_no, prd_no, ot_prd_no, sort_no, regdt, score) \
            select b.gi_cust_no, c.mall_no, c.prd_no, c.ot_prd_no, a.sort_no, sysdate , " + str(limit_size)  + " + 1 - sort_no from BT_DL_CUST_PRD_RANK_D a  \
            inner join dl_cust_map_d b on (a.idx_gi_cust_no = b.idx_gi_cust_no)  \
            inner join dl_prd_map_d c on (a.idx_prd_no = c.idx_prd_no)  \
              "

    curs.execute(sql)
    conn.commit()
    conn.close()


def getMaxUserNum (targetDate):
    conn = getConnection()

    curs = conn.cursor()
    sql = "select max(idx_gi_cust_no) max_cust_no from dl_cust_map_d where dates = '" + targetDate + "'"

    max_user_id = 0
    curs.execute(sql)
    (max_user_id,) = curs.fetchone()

    # conn.close();
    return max_user_id


def getMaxItemNum (targetDate):
    conn = getConnection()

    curs = conn.cursor()
    sql = "select max(idx_prd_no) max_prd_no from dl_prd_map_d where  dates = '" + targetDate + "'"

    max_prd_no = 0
    curs.execute(sql)
    (max_prd_no,) = curs.fetchone()

    # conn.close()
    return max_prd_no



def is_target_user(conn, idx_cust_no, target_date):

    curs = conn.cursor()
    sql = " select count(*) cnt, sum(rec_cnt) rec_cnt from  dl_cust_map_d a \
            inner join dl_cust_target_d b on (a.gi_cust_no = b.gi_cust_no) \
            where a.idx_gi_cust_no = :1  group by a.idx_gi_cust_no "
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

