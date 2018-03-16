from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from batch import dbio




def main():

    dbio.truncate_user_item_rank_a()
    dbio.truncate_user_item_rank_tmp()
    dbio.truncate_dl_cust_prd_info_target()
    dbio.truncate_dl_cust_target()



if __name__ == '__main__':
    main()