import numpy as np
import cx_Oracle
import os
os.putenv('NLS_LANG', '.UTF8')

dbUrl = "db.main.wishlink.info"
dbPort = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1521

def getColorNms ():

    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    sql = "select attr_no||'_'||replace(attr_nm, ' ', '-'), attr_no from ONETEN.ATTR_M where attr_grp_no = 5 and use_yn = 'Y'  order by sort_no  "
    curs.execute(sql)
    cate_nm = curs.fetchall()

    nd = np.array(cate_nm)


    conn.close()
    return nd[:,[0]].flatten(), nd[:,[1]].flatten()



def getColorPrdImgList(attr_no):
    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    # sql = "select  'http://wishimage.styledo.co.kr/RK_46x46/C5_32x32/http://thumb.1ten.co.kr:8110/llbt/700x700/src' || b.img_url, c.filename from \
    sql = "select  'http://1ten-image.wishlink.net/llbt/64x64/src' || b.img_url || '?dummy=20170524', c.filename from \
    			(select min(attr_no) attr_no, ot_prd_no from ONETEN.PRD_ATTR_D where attr_no = " + attr_no + " group by ot_prd_no) at\
                inner join prd_m a on (a.ot_prd_no = at.ot_prd_no)\
                inner join prd_img_d b on (a.ot_prd_no = b.ot_prd_no) \
                inner join item_file c on (b.img_url = c.img_url) \
            where a.prd_use_flag = '02' AND a.prd_disp_flag not in ('01' ) and  mod(a.ot_prd_no,10) > 0  "


    curs.execute(sql)
    img_urls = curs.fetchall()

    nd = np.array(img_urls)

    conn.close()
    if nd.__len__() > 0:
        return nd[:, [0]].flatten(), nd[:, [1]].flatten()
    else:
        return [],[]


def getColorPrdImgListForTest(attr_no):

    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    # sql = "select  'http://wishimage.styledo.co.kr/RK_46x46/C5_32x32/http://thumb.1ten.co.kr:8110/llbt/700x700/src' || b.img_url, c.filename from \
    sql = "select  'http://1ten-image.wishlink.net/llbt/64x64/src' || b.img_url || '?dummy=20170524', c.filename from \
    			(select min(attr_no) attr_no, ot_prd_no from ONETEN.PRD_ATTR_D where attr_no = " + attr_no + " group by ot_prd_no) at\
                inner join prd_m a on (a.ot_prd_no = at.ot_prd_no)\
                inner join prd_img_d b on (a.ot_prd_no = b.ot_prd_no) \
                inner join item_file c on (b.img_url = c.img_url) \
                where a.prd_use_flag = '02' AND a.prd_disp_flag not in ('01' )  and mod(a.ot_prd_no,10) = 0  "


    curs.execute(sql)
    img_urls = curs.fetchall()

    nd = np.array(img_urls)

    conn.close()
    if nd.__len__() > 0:
        return nd[:, [0]].flatten(), nd[:, [1]].flatten()
    else:
        return [],[]

