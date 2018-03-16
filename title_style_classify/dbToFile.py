from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import logging.config
from title_style_classify import dbio
from title_style_classify import common
os.putenv('NLS_LANG', '.UTF8')

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('text_')

def main():
    conn = dbio.getConnection()

    curs = conn.cursor()
    sql= " select p.prd_nm, m.sort_no  from oneten.prd_m p \
        inner join ( \
            select ot_prd_no, min(a.attr_no) attr_no from ONETEN.PRD_ATTR_D a, oneten.attr_m b where a.attr_no = b.attr_no and b.attr_grp_no = 2 group by a.ot_prd_no \
        ) d on (d.ot_prd_no = p.ot_prd_no)  \
        inner join oneten.attr_m m on (m.attr_no = d.attr_no and m.attr_grp_no = 2) \
        where mod(p.ot_prd_no,10) > 0 \
    "

    curs.execute(sql)
    data = curs.fetchall()

    nd = np.array(data)


    curs = conn.cursor()
    sql= " select p.prd_nm, m.sort_no  from oneten.prd_m p \
        inner join ( \
            select ot_prd_no, min(a.attr_no) attr_no from ONETEN.PRD_ATTR_D a, oneten.attr_m b where a.attr_no = b.attr_no and b.attr_grp_no = 2 group by a.ot_prd_no \
        ) d on (d.ot_prd_no = p.ot_prd_no)  \
        inner join oneten.attr_m m on (m.attr_no = d.attr_no and m.attr_grp_no = 2) \
        where mod(p.ot_prd_no,10) = 0 \
    "

    curs.execute(sql)
    data = curs.fetchall()
    nd_test = np.array(data)

    # print(nd)


    with open("/data/www/oneten/title_style_classify/data_dir/data.csv", "w") as f:
        f.write("prd_nm,idx\n")
        # np.savetxt(f, nd, fmt="%s", delimiter=",")
        for data in nd:
            f.write(data[0] + "!@!" +  data[1] +  os.linesep)
        f.close()

    with open("/data/www/oneten/title_style_classify/data_dir/data_test.csv", "w") as f:
        f.write("prd_nm,idx\n")
        # np.savetxt(f, nd, fmt="%s", delimiter=",")
        for data in nd_test:
            f.write(data[0] + "!@!" +  data[1] +  os.linesep)
        f.close()

    # conn.close()


    style_nms, cate_nos = dbio.getStyleNms()
    print(style_nms)
    with open(common.DATA_DIR + "labels.txt", "w") as f:
        for (i, cate_no) in zip(style_nms, cate_nos):
            f.write(i + os.linesep)


if __name__ == '__main__':
    main()