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
    sql = "select  cate_no||'_'||replace(cate_nm, '/', '-') cate_nm, cate_no   from cate_de_d where lrg_cate_no in  (1000000000)  and cate_grp_no = 3   "
    curs.execute(sql)
    cate_nm = curs.fetchall()

    nd = np.array(cate_nm)


    conn.close()
    return nd[:,[0]].flatten(), nd[:,[1]].flatten()




def getCatePrdImgList(cate_no):

    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    sql = "'http://1ten-image.wishlink.net/lrbt/300x300/src' || b.img_url , c.filename \
            from prd_m p \
            inner join prd_img_d b on (p.ot_prd_no = b.ot_prd_no and b.prd_img_type = '02') \
            inner join item_file c on (b.img_url = c.img_url) \
            where p.cate_no = " + cate_no + " and \
             p.prd_use_flag = '02' AND p.prd_disp_flag not in ('01' ) \
            and rownum <= 100 "
    curs.execute(sql)
    img_urls = curs.fetchall()

    nd = np.array(img_urls)

    conn.close()
    if nd.__len__() > 0:
        return nd[:, [0]].flatten(), nd[:, [1]].flatten(), nd[:, [2]].flatten()
    else:
        return [],[],[]

def getPrdPrdImgList():

    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    sql = "select 'http://1ten-image.wishlink.net/lrbt/300x300/src' || b.img_url , c.filename, ca.cate_no||'_'||replace(ca.cate_nm, '/', '-') cate_nm , p.ot_prd_no \
            from prd_m p \
            inner join prd_img_d b on (p.ot_prd_no = b.ot_prd_no and b.prd_img_type = '02') \
            inner join item_file c on (b.img_url = c.img_url) \
            inner join cate_de_d ca on (ca.cate_no = p.cate_no) \
            where \
             p.prd_use_flag = '02' AND p.prd_disp_flag not in ('01' ) \
            "
    curs.execute(sql)
    dataList = curs.fetchall()

    nd = np.array(dataList)

    conn.close()
    if nd.__len__() > 0:
        # return nd[:, [0]].flatten(), nd[:, [1]].flatten(), nd[:, [2]].flatten(), nd[:, [3]].flatten()
        return nd
    else:
        return [],[],[],[]


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
