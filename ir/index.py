#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : index.py
@Time     : 18-12-3 下午12:22
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

from ir.config import Config
from elasticsearch import helpers
import jieba
import json
import pandas as pd
import jieba


class Index(object):
    def __init__(self):
        print("Indexing ...")

    @staticmethod
    def data_convert(file_path="../data/question.csv"):
        print("convert raw csv file into single doc")

        questions = {}
        questions_count = 0
        csv_data = pd.read_csv(file_path, sep='\t', header=None, index_col=0)
        for key, data in csv_data.iterrows():
            if not (data[1].strip() or data[2].strip()):
                continue
            question_cut = ' '.join(token for token in jieba.cut(data[1].strip()))
            questions[questions_count] = {'id': key, 'question': question_cut, 'pid': data[2]}
            questions_count += 1

        return questions

    @staticmethod
    def create_index(config):
        print("creating %s index ..."%config.index_name)
        request_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {
                    "LM": {
                        "type": "LMJelinekMercer",
                        "lambda": 0.4
                    }
                }
            },
            "mapping": {
                config.index_name: {
                    "properties": {
                        "id": {
                            "type": "long",
                            "index": False
                        },
                        "question": {
                            "type": "text",
                            "term_vector": "with_positions_offsets_payloads",
                            "store": True,
                            "analyzer": "standard",
                            "similarity": "LM"
                        },
                        "pid": {
                            "type": "long",
                            "index": False
                        }
                    }
                }
            }
        }

        config.es.indices.delete(index=config.index_name, ignore=[400, 404])
        res = config.es.indices.create(index=config.index_name, body=request_body)
        print(res)
        print("Indices are created successfully")

    @staticmethod
    def bulk_index(questions, bulk_size, config):
        print("Bulk index for question")
        count =1
        actions = []
        for question_count, question in questions.items():
            action = {
                "_index": config.index_name,
                "_type": config.doc_type,
                "_id": question_count,
                "_source": question
            }
            actions.append(action)
            count += 1

            if len(actions) % bulk_size == 0:
                helpers.bulk(config.es, actions)
                print("Bulk index: " + str(count))
                actions = []

        if len(actions) > 0:
            helpers.bulk(config.es, actions)
            print("Bulk index: " + str(count))


def main():
    config = Config()
    index = Index()
    questions = index.data_convert(config.doc_path)
    index.create_index(config)
    index.bulk_index(questions, bulk_size=10000, config=config)


if __name__ == '__main__':
    main()
