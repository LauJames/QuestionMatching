#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : classify_preprocess.py
@Time     : 18-11-28 下午4:29
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""
import pandas as pd
import json

csv_path = './question.csv'
question_label_path = './question_label.txt'
id2label_path = './id2label.json'
primary_question_dict_path = './primary_question_dict_class.json'


def arrange_label_question(csv_path, question_label_path, id2label_path, primary_question_dict_path):
    """
    input: csv files: 问题ID  问题  主问题ID
    transform the raw csv data to label(ID) 问题
    :return: label  问题
    """
    primary_question_dict = {}
    csv_data = pd.read_csv(csv_path, sep='\t', header=None, index_col=0)
    for key, data in csv_data.iterrows():
        if not (data[1].strip() or data[2].strip()):
            continue
        if data[2] == 0:
            primary_question_dict[key] = (data[1]).replace('\n', '')  # key: id, value: question
    primary_question_dict_json = json.dumps(primary_question_dict, ensure_ascii=False)
    with open(primary_question_dict_path, 'w', encoding='utf-8') as pqd_f:
        pqd_f.write(primary_question_dict_json)

    # 构建key: id , value: label
    id2label = {}
    for idx, key in enumerate(primary_question_dict.keys()):
        id2label[key] = idx

    id2label_str = json.dumps(id2label, ensure_ascii=False)
    with open(id2label_path, 'w') as f:
        f.write(id2label_str)

    questions = []
    labels = []
    for key, data in csv_data.iterrows():
        if not (data[1].strip() or data[2].strip()):
            continue
        if data[2] != 0:
            questions.append((data[1]).replace("\n", ""))
            labels.append(id2label[data[2]])
        elif data[2] == 0:
            questions.append((data[1]).replace("\n", ""))
            labels.append(id2label[key])

    with open(question_label_path, 'w', encoding='utf-8') as f:
        for label, question in zip(labels, questions):
            f.write(str(question) + '\t' + str(label) + '\n')


def calc_classes(id2label_path):
    with open(id2label_path, 'r') as f:
        dict_str = f.readline()
        label_dict = json.loads(dict_str)
    print(len(label_dict))  # 1291


if __name__ == '__main__':
    # arrange_label_question(csv_path, question_label_path, id2label_path, primary_question_dict_path)
    calc_classes(id2label_path)
