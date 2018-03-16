# coding=utf-8

import time
from gensim import corpora, models, similarities
import pandas as pd
from collections import defaultdict
from related_prd_by_gensim import dbio
import operator
import threading
import logging.config
import logging
import numpy as np

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('related_prd')


process_start_time = time.time();

def main():


    global ds
    global bow_corpus
    global tfidf
    global ITEM_SIZE
    global thread_result
    global thread_processing_result
    global index

    ds = pd.read_csv("/data/www/oneten/dl_related_prd_by_gensim/prd_txt_processed.csv")
    stoplist = set(''.split(' '))

    raw_corpus = ds['description'].values.tolist()
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in raw_corpus]


    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1


    frequency_sorted = sorted(frequency.items(), reverse=True, key=operator.itemgetter(1))


    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    logger.info("processed_corpus:%s", processed_corpus.__len__())

    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    tfidf = models.TfidfModel(bow_corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=bow_corpus.__len__())

    ITEM_SIZE = 60

    dbio.truncate_related_prd_list_tmp()

    thread_size = 5
    thread_result = np.zeros(thread_size)
    thread_processing_result = np.zeros(thread_size)

    for i in range(thread_size):
        threading.Thread(target=insertData, args=(i, thread_size)).start()

    while 1:
        time.sleep(10)
        logger.info("check thread:[%s]",  thread_result )
        logger.info("thread process cnt:[%s]", thread_processing_result)
        if(thread_result.all() == 1 ):
            logger.info(" thread is end")
            break
        current_time = time.time()
        logger.info("processing time elapsed %s", current_time - process_start_time )

    # for idx, row in ds.iterrows():
    #     data = bow_corpus[idx]
    #     sims = index[tfidf[data]]
    #     similar_indices = sims.argsort()[:-(ITEM_SIZE+2):-1]
    #     dbio.insert_related_prd_list_tmp(row['id'], [ds['id'][i] for i in similar_indices[1:]], conn)


    dbio.truncate_related_prd_list()
    dbio.insert_related_prd_list()



def insertData(thread_num, thread_max):

    conn = dbio.getConnectionByOption(autocommit=True, threaded=True)
    process_cnt = 0
    for idx, row in ds.iterrows():

        if (idx % thread_max == thread_num):
            data = bow_corpus[idx]
            sims = index[tfidf[data]]
            similar_indices = sims.argsort()[:-(ITEM_SIZE+2):-1]
            try:
                dbio.insert_related_prd_list_tmp(row['id'], [ds['id'][i] for i in similar_indices[1:]], conn)
            except Exception as ex:
                logger.error('에러가 발생 했습니다, %s', ex)
            thread_processing_result[thread_num] = thread_processing_result[thread_num] + ITEM_SIZE

    conn.close()
    thread_result[thread_num] = 1

if __name__ == '__main__':
    total_start_time = time.time();

    main()

    total_end_time = time.time();
    logger.info("total elapsed time:%s",total_end_time - total_start_time )
    logger.info("Done!")