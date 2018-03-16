# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
from text_style_classify import dbio
from text_style_classify import common
from text_style_classify import json_parser
from text_style_classify import string_util
import codecs
os.putenv('NLS_LANG', '.UTF8')


def convert(x):
  try:
    return x.astype(int)
  except:
    return x


def writeData():
    conn = dbio.getConnection()

    curs = conn.cursor()
    # sql = "select e.sort_no as idx,  nvl(c.CONTENT_JSON_INFO,'[]')  as text , p.prd_nm from oneten.prd_m p \
    sql = "select e.sort_no as idx, p.prd_nm  \
                , nvl((select WM_CONCAT(keyword_info) from CRAWL.CRL_PRD_KEYWORD where   mall_no = b.mall_no and prd_no = b.prd_no), '') keyword_info \
            , nvl(c.prd_info, ''), m.mall_nm \
            from oneten.prd_m p \
            inner join oneten.crl_prd_d b on (b.OT_PRD_NO = p.ot_prd_no) \
            inner join crawl.crl_prd_list c on (c.mall_no = b.mall_no and c.prd_no = b.prd_no) \
            inner join ( \
    		select ot_prd_no, min(a.attr_no) attr_no from ONETEN.PRD_ATTR_D a, oneten.attr_m b where a.attr_no = b.attr_no and b.attr_grp_no = 2 group by a.ot_prd_no \
    	    ) d on (d.ot_prd_no = p.ot_prd_no) \
            inner join oneten.attr_m e on (e.attr_no = d.attr_no and e.attr_grp_no = 2) \
            inner join oneten.mall_m m on (m.mall_no = p.mall_no) \
            where  mod(p.ot_prd_no,10) > 0 \
        "

    curs.execute(sql)
    data = curs.fetchall()

    nd = np.array(data)
    # for i in range(len(nd)):
        # nd[i][1] = unicode(nd[i][2]) + ' ' + json_parser.getText(nd[i][1].read()).replace("\n", " ")
        # nd[i][1] = nd[i][2] + ' ' + json_parser.getText(nd[i][1].read()).replace("\n", " ").encode('utf-8')
        # nd[i][1] = nd[i][2] + ' ' + json_parser.getText(nd[i][1].read()).replace("\n", " ").encode('utf-8')
        # nd[i][1] = nd[i][1] + ' ' +  nd[i][2]


    with codecs.open("/data/www/oneten/text_style_classify/data_dir/data.csv", "w") as f:
        f.write("text,idx\n")
        # np.savetxt(f, nd, fmt="%s", delimiter=",")
        for data in nd:
            # text = data[1] + ' ' + data[2] + ' ' + data[3];
            # text = data[1] + ' ' + data[2];
            # text = data[1]
            # text = text.replace("\n", " ").replace("!", "").strip()
            text = string_util.getFilteredText(prd_nm=data[1], keyword_info= data[2], prd_info= data[3], mall_nm = data[4])
            f.write(text + "!@!" + str(data[0]) + os.linesep)
        f.close()


def writeTestData():
    conn = dbio.getConnection()

    curs = conn.cursor()
    sql = "select e.sort_no as idx, p.prd_nm  \
                , nvl((select WM_CONCAT(keyword_info) from CRAWL.CRL_PRD_KEYWORD where   mall_no = b.mall_no and prd_no = b.prd_no), '') keyword_info \
            , nvl(c.prd_info, ''), m.mall_nm \
            from oneten.prd_m p \
            inner join oneten.crl_prd_d b on (b.OT_PRD_NO = p.ot_prd_no) \
            inner join crawl.crl_prd_list c on (c.mall_no = b.mall_no and c.prd_no = b.prd_no) \
            inner join ( \
    		select ot_prd_no, min(a.attr_no) attr_no from ONETEN.PRD_ATTR_D a, oneten.attr_m b where a.attr_no = b.attr_no and b.attr_grp_no = 2 group by a.ot_prd_no \
    	    ) d on (d.ot_prd_no = p.ot_prd_no) \
            inner join oneten.attr_m e on (e.attr_no = d.attr_no and e.attr_grp_no = 2) \
            inner join oneten.mall_m m on (m.mall_no = p.mall_no) \
            where  mod(p.ot_prd_no,10) = 0 and e.sort_no = 1 \
        "

    curs.execute(sql)
    data = curs.fetchall()

    nd = np.array(data)
    # for i in range(len(nd)):
    #     # nd[i][1] = nd[i][2] + ' ' + json_parser.getText(nd[i][1].read()).replace("\n", " ").encode('utf-8')
    #     # nd[i][1] = nd[i][2] + ' ' + json_parser.getText(nd[i][1].read()).replace("\n", " ").encode('utf-8')
    #     text = str(data[1]) + ' ' + str(data[2]) + ' ' + str(data[3]);
    #     text = text.replace("\n", " ").replace("!", "")


    with codecs.open("/data/www/oneten/text_style_classify/data_dir/data_test.csv", "w") as f:
        f.write("text,idx\n")
        # np.savetxt(f, nd, fmt="%s", delimiter=",")
        for data in nd:
            # f.write(data[1] + "!@!" + str(data[0]) + os.linesep)
            # text = data[1] + ' ' + data[2] + ' ' + data[3];
            #text = data[1] + ' ' + data[2];
            # text = data[1]
            # text = text.replace("\n", " ").replace("!", "").strip()
            text = string_util.getFilteredText(prd_nm=data[1], keyword_info= data[2], prd_info= data[3], mall_nm = data[4])
            f.write(text + "!@!" + str(data[0]) + os.linesep)
        f.close()
    # conn.close()
writeData()
writeTestData()


#
# cate_nms, cate_nos = dbio.getCateNms()
# print(cate_nms)
# with open(common.DATA_DIR + "labels.txt", "w") as f:
#     for (i, cate_no) in zip(cate_nms, cate_nos):
#         f.write(i + os.linesep)
