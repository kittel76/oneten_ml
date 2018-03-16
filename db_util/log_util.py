import cx_Oracle
import time
import os
os.putenv('NLS_LANG', '.UTF8')
dbUrl = "db.main.wishlink.info"
dbPort = 1521

if os.getenv("pythonAppType", "") == "local":
    dbUrl = "hostway.gate.wishlink.info"
    dbPort = 1521


def update_job_log   (dates=""  , job_code="", batch_size=0, data_cnt=0, traing_step=0, accuracy_train=0.0, accuracy_test=0.0):

    now = time.localtime()
    if(dates==""):
        dates = str(now.tm_year) + str(now.tm_mon).rjust(2,'0') + str(now.tm_mday).rjust(2,'0')

    print("dates", dates)
    print("job_code", job_code)
    print("data_cnt", data_cnt)
    print("traing_step", traing_step)
    print("accuracy_train", accuracy_train)
    print("accuracy_test", accuracy_test)
    print("batch_size", batch_size)

    if accuracy_test==0:
        print("accuracy_test zero")



    conn = cx_Oracle.connect("stat", "stat2017#!", cx_Oracle.makedsn(dbUrl, dbPort, "oraST1"))
    curs = conn.cursor()
    str_list = []
    str_list.append(" merge into ml_batch_job_log ")
    str_list.append(" 	using dual on (dates = '" + dates + "' and job_code = '" + job_code + "' )")
    str_list.append(" 	when matched then")
    str_list.append(" 		update set")
    str_list.append(" 		    updt = sysdate")
    if data_cnt > 0:
        str_list.append(" 			, data_cnt = " + str(data_cnt))
    if traing_step > 0:
        str_list.append(" 			, traing_step = " + str(traing_step))
    if accuracy_train > 0:
        str_list.append(" 			, accuracy_train = " + str(accuracy_train))
    if accuracy_test > 0:
        str_list.append(" 			, accuracy_test = " + str(accuracy_test))
    if batch_size > 0:
        str_list.append(" 			, batch_size = " + str(batch_size))
    str_list.append(" 	when not matched then")
    str_list.append(" 		insert (dates, job_code, data_cnt, traing_step, accuracy_train, accuracy_test, regdt) values")
    str_list.append(" 		(")
    str_list.append(" 			'" + dates + "', '" + job_code + "'" +"," + str(data_cnt) +"," + str(traing_step) +"," + str(accuracy_train) +"," + str(accuracy_test) + ", sysdate")
    str_list.append(" 		)")
    sql = ''.join(str_list)

    curs.execute(sql)

    str_list = []
    str_list.append(" insert into ml_batch_job_log_hist   (dates, job_code, data_cnt, traing_step, accuracy_train, accuracy_test, regdt) values  ")
    str_list.append(" 		(")
    str_list.append(" 			'" + dates + "', '" + job_code + "'" +"," + str(data_cnt) +"," + str(traing_step) +"," + str(accuracy_train) +"," + str(accuracy_test) + ", sysdate")
    str_list.append(" 		)")
    sql = ''.join(str_list)
    print(sql)
    curs.execute(sql)

    conn.commit()