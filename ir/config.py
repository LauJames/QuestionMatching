#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : config.py
@Time     : 18-12-3 下午12:13
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

from elasticsearch import Elasticsearch


class Config(object):
    def __init__(self):
        print("config...")
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        self.index_name = "q2qmatch"
        self.doc_type = "questions"

        file_path = '../data/question.csv'
        self.doc_path = file_path


def main():
    Config()


if __name__ == '__main__':
    main()
