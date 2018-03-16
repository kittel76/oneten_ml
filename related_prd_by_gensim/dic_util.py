#-*- coding: utf-8 -*-
import pandas as pd
from collections import defaultdict
from konlpy.tag import Kkma
import re
import sys
reload(sys)
sys.setdefaultencoding('UTF8')


kkma = Kkma()

def getDictionary(fine_name):
    ds = pd.read_csv("./dic.txt")
    synonym_list = ds['synonym'].values.tolist()
    dict = defaultdict()
    for idx, document in enumerate(synonym_list):
        for sy in document.split(';'):
            dict[sy] =  ds['word'][idx]

    return dict


def getSynonymApplyString(src, dict):
    ret = []
    word = ' '.join(kkma.sentences(src ))
    for data in re.split("[-_+ ]+", word):
        if data.encode('utf-8').lower() in dict:
            ret.append(dict[data.encode('utf-8').lower()])
        else:
            ret.append(data)
    return ' '.join(ret)

if __name__ == '__main__':

    dict = getDictionary("./dic.txt")

    #
    tmp = "멜로 우 파스텔 기모 반목 T 밴딩 스커트 세트"
    # # tmp = "오프숄더"
    # tmp = " 사각"
    #
    # print(tmp)
    print(getSynonymApplyString(tmp, dict))


