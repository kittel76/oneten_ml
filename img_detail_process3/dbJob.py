import numpy as np
import cx_Oracle
import os
from img_detail_process3 import common
import tensorflow as tf
os.putenv('NLS_LANG', '.UTF8')

FLAGS = tf.app.flags.FLAGS
dbUrl = FLAGS.dbUrl
dbPort = FLAGS.dbPort

def getStyleNms ():
    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    sql = "select attr_no||'_'||attr_nm, attr_no from ONETEN.ATTR_M where attr_grp_no = 6 order by sort_no  "
    curs.execute(sql)
    cate_nm = curs.fetchall()

    nd = np.array(cate_nm)


    conn.close()
    return nd[:,[0]].flatten(), nd[:,[1]].flatten()



def getStylePrdImgList(style_no):

    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    # sql = "select  'http://wishimage.styledo.co.kr/RK_46x46/C5_32x32/http://thumb.1ten.co.kr:8110/llbt/700x700/src' || b.img_url, c.filename from \
    sql = "select  'http://thumb.1ten.co.kr:8110/llbt/" + FLAGS.image_size + "/src' || b.img_url || '?dummy=20170524', c.filename from \
    			(select min(attr_no) attr_no, ot_prd_no from ONETEN.PRD_ATTR_D where attr_no = " + style_no + " group by ot_prd_no) at\
                inner join prd_m a on (a.ot_prd_no = at.ot_prd_no)\
                inner join prd_img_d b on (a.ot_prd_no = b.ot_prd_no) \
                inner join item_file c on (b.img_url = c.img_url) \
                where  mod(a.ot_prd_no,10) > 0    "



    curs.execute(sql)
    img_urls = curs.fetchall()

    nd = np.array(img_urls)

    conn.close()
    if nd.__len__() > 0:
        return nd[:, [0]].flatten(), nd[:, [1]].flatten()
    else:
        return [],[]


def getStylePrdImgListForTest (style_no):

    conn = cx_Oracle.connect("oneten", "oneten2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraOT1"))

    curs = conn.cursor()
    sql = "select  'http://thumb.1ten.co.kr:8110/llbt/" + FLAGS.image_size + "/src' || b.img_url || '?dummy=20170524', c.filename from \
    			(select min(attr_no) attr_no, ot_prd_no from ONETEN.PRD_ATTR_D where attr_no = " + style_no + " group by ot_prd_no) at\
                inner join prd_m a on (a.ot_prd_no = at.ot_prd_no)\
                inner join prd_img_d b on (a.ot_prd_no = b.ot_prd_no) \
                inner join item_file c on (b.img_url = c.img_url) \
                where  mod(a.ot_prd_no,10) = 0     "

    curs.execute(sql)
    img_urls = curs.fetchall()

    nd = np.array(img_urls)

    conn.close()
    if nd.__len__() > 0:
        return nd[:, [0]].flatten(), nd[:, [1]].flatten()
    else:
        return [],[]


