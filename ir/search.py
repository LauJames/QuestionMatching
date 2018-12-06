#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : search.py.py
@Time     : 18-12-4 下午3:50
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

from ir.config import Config


class Search(object):
    def __init__(self):
        print("Searching ...")

    @staticmethod
    def search_by_question(question, top_n, config):
        q = {
            "query": {
                "multi_match": {
                    "query": question,
                    "fields": ["question"],
                    "fuzziness": "AUTO"
                }
            }
        }

        count = 0
        while count < top_n:
            try:
                res = config.es.search(index=config.index_name, doc_type=config.doc_type, body=q, request_timeout=30)
                topn = res['hits']['hits']
                count = 0
                result = []
                for data in topn:
                    if count < top_n:
                        result.append((data['_source']['id'], data['_source']['question'], data['_source']['pid']))
                        count += 1
                return result
            except Exception as e:
                print(e)
                print("Try again!")
                count += 1
                continue

        print("ReadTimeOutError may not be covered!")
        return []


def main():
    config = Config()
    search = Search()
    query = "保险怎么买？"
    result = search.search_by_question(query, 5, config)
    # for data in result:
    #     print(data[0], data[1], data[2])
    print(result)


if __name__ == '__main__':
    main()
