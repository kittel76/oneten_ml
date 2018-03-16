

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

def main():


    global ds
    global bow_corpus
    global tfidf
    global ITEM_SIZE
    global thread_result
    global index

    ds = pd.read_csv("/data/www/oneten/dl_related_prd_by_gensim/prd_txt.csv")
    stoplist = set(''.split(' '))

    raw_corpus = ds['description'].values.tolist()
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in raw_corpus]


    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1


    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    logger.info("processed_corpus:%s", processed_corpus.__len__())

    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    tfidf = models.TfidfModel(bow_corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=bow_corpus.__len__())

    print("bow_corpus[idx]", bow_corpus[1000])





    doc = "플라워 스커트"
    vec_bow = dictionary.doc2bow(doc.lower().split())


    print("vec_bow", vec_bow)

    ITEM_SIZE = 30

    sims = index[tfidf[vec_bow]]
    similar_indices = sims.argsort()[:-(ITEM_SIZE + 2):-1]

    print("similar_indices", similar_indices)

    for i in similar_indices:
        print(ds['id'][i], ds["description"][i], i)

    # for idx, row in ds.iterrows():
    #     data = bow_corpus[idx]
    #     sims = index[tfidf[data]]
    #     similar_indices = sims.argsort()[:-(ITEM_SIZE+2):-1]
    #     dbio.insert_related_prd_list_tmp(row['id'], [ds['id'][i] for i in similar_indices[1:]], conn)






def insertData(thread_num, thread_max):

    conn = dbio.getConnectionByOption(autocommit=True, threaded=True)
    for idx, row in ds.iterrows():

        if (idx % thread_max == thread_num):
            data = bow_corpus[idx]
            sims = index[tfidf[data]]
            similar_indices = sims.argsort()[:-(ITEM_SIZE+2):-1]
            dbio.insert_related_prd_list_tmp(row['id'], [ds['id'][i] for i in similar_indices[1:]], conn)

    conn.close()
    thread_result[thread_num] = 1

if __name__ == '__main__':
    total_start_time = time.time();

    main()

    total_end_time = time.time();
    logger.info("total elapsed time:%s",total_end_time - total_start_time )
    logger.info("Done!")