#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : wordCount.py
@Time     : 2018/11/19 14:41
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""
import numpy as np
import pandas as pd
import codecs
from collections import Counter


def contentlen():
    with open("question", 'r', encoding='utf-8') as f:
        lenList = []
        for line in f:
            dframe = line.split('\t')
            content = dframe[1]
            lenList.append(len(content))
        pdstata(lenList)


def pdstata(lenList):
    c = Counter()
    for num in lenList:
        c[num] = c[num] + 1
    len_pd = pd.Series(lenList)
    meanList = len_pd.mean()
    maxList = len_pd.max()
    minList = len_pd.min()
    medianList = len_pd.median()
    countList = len_pd.count()
    quantileList = len_pd.quantile([0.25, 0.75])
    try:
        f = codecs.open('stats', 'w+', 'utf-8')
        f.write('共有' + str(countList) + '条数据 \r\n')
        f.write('长度频次统计列表：\r\n')
        f.write(str(c))
        f.write('长度均值:' + str(meanList) + '\r\n')
        f.write('长度最大值:' + str(maxList) + '\r\n')
        f.write('长度最小值:' + str(minList) + '\r\n')
        f.write('长度中位数:' + str(medianList) + '\r\n')
        f.write('1/4分位数、3/4分位数：')
        f.write(str(quantileList) + '\r\n')
    finally:
        f.close()
