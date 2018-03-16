import numpy as np
from batch import dbio

# a = np.array([True, False])

conn = dbio.getConnectionByOption(autocommit=False, threaded=True)
is_target_user, rec_cnt = dbio.is_target_user_thread(conn, 0)


print("is_target_user, rec_cnt", is_target_user, rec_cnt)