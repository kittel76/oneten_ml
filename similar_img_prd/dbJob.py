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
    sql = "select  cate_no||'_'||replace(cate_nm, '/', '-') cate_nm, cate_no   from cate_de_d where cate_grp_no = 3   "
    curs.execute(sql)
    cate_nm = curs.fetchall()

    nd = np.array(cate_nm)


    conn.close()
    return nd[:,[0]].flatten(), nd[:,[1]].flatten()




def getCatePrdImgList(cate_no):

    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    sql = "select  'http://thumb.1ten.co.kr:8110/lrbt/300x300/src/' || b.img_url , p.ot_prd_no||'.jpg' from \
            prd_m p \
            inner join prd_img_d b on (p.ot_prd_no = b.ot_prd_no and b.prd_img_type = '02') \
            inner join item_file c on (b.img_url = c.img_url) \
            where p.cate_no = " + cate_no + " and \
            p.prd_disp_flag = '02' and p.prd_use_flag = '02' \
            and SYSDATE BETWEEN  p.disp_start_dtm AND p.disp_end_dtm \
            and rownum <= 1000000 "
    curs.execute(sql)
    img_urls = curs.fetchall()

    nd = np.array(img_urls)

    conn.close()
    if nd.__len__() > 0:
        return nd[:, [0]].flatten(), nd[:, [1]].flatten()
    else:
        return [],[]


def getImg(fileNo):
    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    print("fileNo", fileNo)
    curs = conn.cursor()
    sql = "select  'http://1ten-image.wishlink.net/'|| img_url  from item_file where file_no =  " +  str(fileNo)
    curs.execute(sql)
    cate_nm = curs.fetchone()

    print("cate_nm", cate_nm[0])

    conn.close()
    return cate_nm[0]
