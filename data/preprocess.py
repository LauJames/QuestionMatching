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
import json


excel_path = './question.xls'
question_bd = './paragraph_column.txt'
qq_path_bd = 'q2q_pair_bd.txt'
qq_path_tk = './q2q_pair_tk.txt'
question_tk = './question.csv'
test_path = './testset.txt'
primary_question_dict_path_tk = './primary_question_dict_tk.json'
primary_question_dict_path_bd = './primary_question_dict_bd.json'
front_noise = './front_noise.txt'
end_noise = './end_noise.txt'


def load_noise(front_path, end_path):
    """
    Load the front and end noise
    :param front_path:
    :param end_path:
    :return: List: front_list, end_list
    """
    front_list = []
    end_list = []
    with open(front_path, 'r', encoding='utf-8') as front_f:
        while True:
            line = front_f.readline()
            if not line:
                print('Front noise phrase load finished!')
                break
            front_list.append(line.replace('\n', ''))

    with open(end_path, 'r', encoding='utf-8') as end_f:
        while True:
            line = end_f.readline()
            if not line:
                print('End noise phrase load finished!')
                return front_list, end_list
            end_list.append(line.replace('\n', ''))


def txt2QQpair_tune(path, out_path, front_path='./front_noise.txt', end_path='./end_noise.txt'):
    """
    input: csv files: 问题ID  问题  主问题ID
    transform the txt data to Question-Question pairs
    :return: Tag 问题 问题
    """
    # load noise prefix and end_fix
    front_list, end_list = load_noise(front_path, end_path)

    # load primary question
    primary_question_dict = {}
    with open(path, 'r', encoding='utf-8') as bd_f:
        while True:
            line = bd_f.readline()
            if not line:
                print('Primary question dict construct successfully!')
                break
            try:
                temp_data = (line.replace('\n', '')).strip().split('\t')
                if len(temp_data) != 3:
                    continue
                temp_id = int(temp_data[0])
                temp_context = temp_data[1]
                temp_pid = int(temp_data[2])
                if not temp_context.strip():
                    continue
                if temp_pid == 0:
                    primary_question_dict[temp_id] = temp_context.replace('\n', '')  # key: id, value: question
            except Exception as e:
                print(line)
                print(e)

    primary_question_dict_json = json.dumps(primary_question_dict, ensure_ascii=False)
    with open(primary_question_dict_path_bd, 'w', encoding='utf-8') as pqd_f:
        pqd_f.write(primary_question_dict_json)
    # end of load primary question

    # construct question to question pair

    questions1 = []
    questions2 = []
    flags = []

    with open(path, 'r', encoding='utf-8') as bd_f:
        while True:
            line = bd_f.readline()
            # print(len(flags))
            if len(flags) >= 200000:
                break
            if not line:
                print('question to question matching data construct successfully')
                break
            try:
                temp_data = (line.replace('\n', '')).strip().split('\t')
                if len(temp_data) != 3:
                    continue
                temp_id = int(temp_data[0])
                temp_context = temp_data[1]
                if len(flags) < 150000:
                    temp_context_noise = random.choice(front_list) + temp_context + random.choice(end_list)
                else:
                    temp_context_noise = temp_context
                temp_pid = int(temp_data[2])
                if not temp_context.strip():
                    continue
                if temp_pid != 0:
                    # questions1.append((temp_context.replace('\n', '')))
                    questions1.append((temp_context_noise.replace('\n', '')))
                    questions2.append(primary_question_dict[temp_pid])
                    flags.append(1)

                    # add unnoise data
                    questions1.append((temp_context.replace('\n', '')))
                    questions2.append(primary_question_dict[temp_pid])
                    flags.append(1)

                    temp_dict = primary_question_dict.copy()
                    primary_id_raw = list(temp_dict.keys())

                    # negative sample: 2:2
                    primary_id_raw.remove(temp_pid)
                    fake_id = random.choice(primary_id_raw)
                    questions1.append(temp_context.replace('\n', ''))
                    questions2.append(primary_question_dict[fake_id])
                    flags.append(0)

                    primary_id_raw.remove(fake_id)
                    fake_id = random.choice(primary_id_raw)
                    questions1.append(temp_context.replace('\n', ''))
                    questions2.append(primary_question_dict[fake_id])
                    flags.append(0)

                    # primary_id_raw.remove(fake_id)
                    # fake_id = random.choice(primary_id_raw)
                    # questions1.append(temp_context.replace('\n', ''))
                    # questions2.append(primary_question_dict[fake_id])
                    # flags.append(0)

            except Exception as e:
                print(line)
                print(e)

    with codecs.open(out_path, 'w', encoding='utf-8') as qq:
        for flag, q1, q2 in zip(flags, questions1, questions2):
            if q1 and q2:
                qq.write(str(flag) + '\t' + str(q1) + '\t' + str(q2) + '\n')


def csv2QQpair_tune(path, out_path, front_path='./front_noise.txt', end_path='./end_noise.txt'):
    """
    input: csv files: 问题ID  问题  主问题ID
    transform the csv data to Question-Question pairs
    :return: Tag 问题 问题
    """
    # load noise prefix and end_fix
    front_list, end_list = load_noise(front_path, end_path)

    primary_question_dict = {}
    # suitable for csv file only
    csv_data = pd.read_csv(path, sep='\t', header=None, index_col=0)
    for key, data in csv_data.iterrows():
        if not (data[1].strip() or data[2].strip()):
            continue
        if data[2] == 0:
            primary_question_dict[key] = (data[1]).replace('\n', '')  # key: id, value: question

    primary_question_dict_json = json.dumps(primary_question_dict, ensure_ascii=False)
    with open(primary_question_dict_path_tk, 'w', encoding='utf-8') as pqd_f:
        pqd_f.write(primary_question_dict_json)

    questions1 = []
    questions2 = []
    flags = []

    # suitable for csv file only
    for key, data in csv_data.iterrows():
        if not (data[1].strip() or data[2].strip()):
            continue
        if data[2] != 0:
            temp_context = (data[1]).replace("\n", "")
            if len(flags) < 100000:
                temp_context_noise = random.choice(front_list) + temp_context + random.choice(end_list)
            else:
                temp_context_noise = temp_context
            # True
            questions1.append(temp_context_noise)
            questions2.append(primary_question_dict[data[2]])
            flags.append(1)

            # add unnoise data
            questions1.append((temp_context.replace('\n', '')))
            questions2.append(primary_question_dict[data[2]])
            flags.append(1)

            temp_dict = primary_question_dict.copy()  # 浅拷贝，避免修改主问题列表
            # dict.keys() 返回dict_keys类型，其性质类似集合(set)而不是列表(list)，因此不能使用索引获取其元素
            primary_id_raw = list(temp_dict.keys())

            # negative sample ratio: 2:2
            primary_id_raw.remove(data[2])  # 先去除该问题主问题id，再随机负采样
            fake_id = random.choice(primary_id_raw)

            questions1.append(temp_context)
            questions2.append(primary_question_dict[fake_id])
            flags.append(0)

            primary_id_raw.remove(fake_id)
            fake_id = random.choice(primary_id_raw)
            questions1.append(temp_context)
            questions2.append(primary_question_dict[fake_id])
            flags.append(0)

            # primary_id_raw.remove(fake_id)
            # fake_id = random.choice(primary_id_raw)
            # questions1.append(temp_context)
            # questions2.append(primary_question_dict[fake_id])
            # flags.append(0)

    with codecs.open(out_path, 'w', encoding='utf-8') as qq:
        for flag, q1, q2 in zip(flags, questions1, questions2):
            if q1 and q2:
                qq.write(str(flag) + '\t' + str(q1) + '\t' + str(q2) + '\n')

def excel2csv(path, out_path):
    """
    Multi-Sheets excel file, needs to be convert to one file
    :param path: str
    :param out_path: str
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
    csv_df.to_csv(out_path, sep='\t', header=0)


def txt2QQpair(path, out_path):
    """
        input: csv files: 问题ID  问题  主问题ID
        transform the txt data to Question-Question pairs
        :return: Tag 问题 问题
        """
    primary_question_dict = {}
    with open(path, 'r', encoding='utf-8') as bd_f:
        while True:
            line = bd_f.readline()
            if not line:
                print('Primary question dict construct successfully!')
                break
            try:
                temp_data = (line.replace('\n', '')).strip().split('\t')
                if len(temp_data) != 3:
                    continue
                temp_id = int(temp_data[0])
                temp_context = temp_data[1]
                temp_pid = int(temp_data[2])
                if not temp_context.strip():
                    continue
                if temp_pid == 0:
                    primary_question_dict[temp_id] = temp_context.replace('\n', '')  # key: id, value: question
            except Exception as e:
                print(line)
                print(e)

    primary_question_dict_json = json.dumps(primary_question_dict, ensure_ascii=False)
    with open(primary_question_dict_path_bd, 'w', encoding='utf-8') as pqd_f:
        pqd_f.write(primary_question_dict_json)

    questions1 = []
    questions2 = []
    flags = []

    with open(path, 'r', encoding='utf-8') as bd_f:
        while True:
            line = bd_f.readline()
            print(len(flags))
            if len(flags) >= 200000:
                break
            if not line:
                print('question to question matching data construct successfully')
                break
            try:
                temp_data = (line.replace('\n', '')).strip().split('\t')
                if len(temp_data) != 3:
                    continue
                temp_id = int(temp_data[0])
                temp_context = temp_data[1]
                temp_pid = int(temp_data[2])
                if not temp_context.strip():
                    continue
                if temp_pid != 0:
                    questions1.append((temp_context.replace('\n', '')))
                    questions2.append(primary_question_dict[temp_pid])
                    flags.append(1)
                    temp_dict = primary_question_dict.copy()
                    primary_id_raw = list(temp_dict.keys())
                    primary_id_raw.remove(temp_pid)
                    fake_id = random.choice(primary_id_raw)
                    questions1.append(temp_context.replace('\n', ''))
                    questions2.append(primary_question_dict[fake_id])
                    flags.append(0)

            except Exception as e:
                print(line)
                print(e)

    with codecs.open(out_path, 'w', encoding='utf-8') as qq:
        for flag, q1, q2 in zip(flags, questions1, questions2):
            if q1 and q2:
                qq.write(str(flag) + '\t' + str(q1) + '\t' + str(q2) + '\n')


def csv2QQpair(path, out_path):
    """
    input: csv files: 问题ID  问题  主问题ID
    transform the csv data to Question-Question pairs
    :return: Tag 问题 问题
    """
    primary_question_dict = {}
    # suitable for csv file only
    csv_data = pd.read_csv(path, sep='\t', header=None, index_col=0)
    for key, data in csv_data.iterrows():
        if not (data[1].strip() or data[2].strip()):
            continue
        if data[2] == 0:
            primary_question_dict[key] = (data[1]).replace('\n', '')  # key: id, value: question

    primary_question_dict_json = json.dumps(primary_question_dict, ensure_ascii=False)
    with open(primary_question_dict_path_tk, 'w', encoding='utf-8') as pqd_f:
        pqd_f.write(primary_question_dict_json)

    questions1 = []
    questions2 = []
    flags = []

    # suitable for csv file only
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

    with codecs.open(out_path, 'w', encoding='utf-8') as qq:
        for flag, q1, q2 in zip(flags, questions1, questions2):
            if q1 and q2:
                qq.write(str(flag) + '\t' + str(q1) + '\t' + str(q2) + '\n')


def gen_testset(path):
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
    count = 0
    for key, data in csv_data.iterrows():
        if count > 1000:
            break
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
            count += 1

    with codecs.open(test_path, 'w', encoding='utf-8') as qq:
        for flag, q1, q2 in zip(flags, questions1, questions2):
            if q1 and q2:
                qq.write(str(flag) + '\t' + str(q1) + '\t' + str(q2) + '\n')


if __name__ == '__main__':
    # csv2QQpair(question_tk, qq_path_tk)
    # txt2QQpair(question_bd, qq_path_bd)
    # csv2QQpair_tune(question_tk, qq_path_tk)
    txt2QQpair_tune(question_bd, qq_path_bd)

