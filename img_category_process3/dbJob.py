import numpy as np
import cx_Oracle
import os
os.putenv('NLS_LANG', '.UTF8')

dbUrl = "db.main.wishlink.info"
dbPort = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1521


def getCateNms ():
    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    sql = "select  cate_no||'_'||replace(cate_nm, '/', '-') cate_nm, cate_no   from cate_de_d where lrg_cate_no in  (1000000000, 2000000000)  and cate_grp_no = 3   "
    curs.execute(sql)
    cate_nm = curs.fetchall()

    nd = np.array(cate_nm)


    conn.close()
    return nd[:,[0]].flatten(), nd[:,[1]].flatten()


def getCateNmsForTest ():
    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    sql = "select  cate_no||'_'||replace(cate_nm, '/', '-') cate_nm, cate_no   from cate_de_d where lrg_cate_no in  (1000000000, 2000000000)  and cate_grp_no = 3   "
    curs.execute(sql)
    cate_nm = curs.fetchall()

    nd = np.array(cate_nm)


    conn.close()
    return nd[:,[0]].flatten(), nd[:,[1]].flatten()



def getCatePrdImgList(cate_no):

    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    sql = "select  'http://thumb.1ten.co.kr:8110/llbt/128x128/src/' || b.img_url || '?dummy=20170524', c.filename from \
            prd_m a \
            inner join prd_img_d b on (a.ot_prd_no = b.ot_prd_no) \
            inner join item_file c on (b.img_url = c.img_url) \
            where a.cate_no = " + cate_no  + " and mod(a.ot_prd_no,20) > 0 and  rownum <= 10000000"
    curs.execute(sql)
    img_urls = curs.fetchall()

    nd = np.array(img_urls)

    conn.close()
    if nd.__len__() > 0:
        return nd[:, [0]].flatten(), nd[:, [1]].flatten()
    else:
        return [],[]



def getCatePrdImgListForTest(cate_no):

    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    # sql = "select  'http://wishimage.styledo.co.kr/RK_46x46/C5_32x32/http://thumb.1ten.co.kr:8110/llbt/700x700/src' || b.img_url, c.filename from \
    sql = "select  'http://thumb.1ten.co.kr:8110/llbt/128x128/src/' || b.img_url || '?dummy=20170524', c.filename from \
            prd_m a \
            inner join prd_img_d b on (a.ot_prd_no = b.ot_prd_no) \
            inner join item_file c on (b.img_url = c.img_url) \
            where a.cate_no = " + cate_no  + " and mod(a.ot_prd_no,20) = 0 and rownum <= 100000000 "
    curs.execute(sql)
    img_urls = curs.fetchall()

    nd = np.array(img_urls)

    conn.close()
    if nd.__len__() > 0:
        return nd[:, [0]].flatten(), nd[:, [1]].flatten()
    else:
        return [],[]


