#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : preprocess.py
@Time     : 18-11-13 上午11:14
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import pandas as pd
import random
import codecs


excel_path = './question.xls'
csv_path = './question.csv'
qq_path = './q2q_pair.txt'


def excel2csv(path):
    """
    Multi-Sheets excel file, needs to be convert to one file
    :param path: str
    :return:
    """
    io = pd.io.excel.ExcelFile(path)
    excel_data = pd.read_excel(io,
                               sheet_name=['question', 'question(2)', 'question(3)'],
                               # sheet_name=['Sheet1', 'Sheet2', 'Sheet3'],
                               usecols=[0, 1, 2],  # 0: id,  1: question, 2:parent_id
                               index_col=0,
                               header=None)
    csv_df = pd.concat([excel_data['question'], excel_data['question(2)'], excel_data['question(3)']])
    csv_df.to_csv(csv_path, sep='\t', header=0)


def csv2QQpair(path):
    """
    input: csv files: 问题ID  问题  主问题ID
    transform the csv data to Question-Question pairs
    :return: Tag 问题 问题
    """
    primary_question_dict = {}
    csv_data = pd.read_csv(path, sep='\t', header=None, index_col=0)
    for key, data in csv_data.iterrows():
        if not (data[1].strip() or data[2].strip()):
            continue
        if data[2] == 0:
            primary_question_dict[key] = (data[1]).replace('\n', '')  # key: id, value: question

    questions1 = []
    questions2 = []
    flags = []
    for key, data in csv_data.iterrows():
        if not (data[1].strip() or data[2].strip()):
            continue
        if data[2] != 0:
            # True
            questions1.append((data[1]).replace("\n", ""))
            questions2.append(primary_question_dict[data[2]])
            flags.append(1)
            temp_dict = primary_question_dict.copy()  # 浅拷贝，避免修改主问题列表
            # dict.keys() 返回dict_keys类型，其性质类似集合(set)而不是列表(list)，因此不能使用索引获取其元素
            primary_id_raw = list(temp_dict.keys())
            primary_id_raw.remove(data[2])  # 先去除该问题主问题id，再随机负采样
            fake_id = random.choice(primary_id_raw)

            questions1.append((data[1]).replace("\n", ""))
            questions2.append(primary_question_dict[fake_id])
            flags.append(0)

    with codecs.open(qq_path, 'w', encoding='utf-8') as qq:
        for flag, q1, q2 in zip(flags, questions1, questions2):
            if q1 and q2:
                qq.write(str(flag) + '\t' + str(q1) + '\t' + str(q2) + '\n')


if __name__ == '__main__':
    csv2QQpair(csv_path)
