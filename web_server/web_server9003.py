#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : web_server9003.py
@Time     : 18-11-28 上午11:02
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import tornado.ioloop
import tornado.web
import json
import os
import sys
import jieba
import numpy as np
from ir.search import Search
from ir.config import Config

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

from infer import prepare, inference, infer_prob


def chinese_tokenizer(documents):
    """
    中文文本转换为词序列(restore时还需要用到，必须包含)
    :param documents:
    :return:
    """
    for document in documents:
        yield list(jieba.cut(document))


vocab_processor, model, session = prepare()


class MatchHandler(tornado.web.RequestHandler):

    def data_received(self, chunk):
        pass

    def get(self, *args, **kwargs):
        self.render('index.html')

    def post(self, *args, **kwargs):
        self.use_write()

    def use_write(self):
        q1 = [self.get_argument('q1')]
        q2 = [self.get_argument('q2')]
        try:
            tag = inference(q1, q2, vocab_processor, model, session)
            json_data = {'question1': str(q1),
                         'question2': str(q2),
                         'match': True if tag else False}
            print(json_data)
            self.write(json.dumps(json_data, ensure_ascii=False))
        except Exception as e:
            print(e)
            json_data = {'question1': str(q1),
                         'question2': str(q2),
                         'match': 'Unknown'}
            self.write(json.dumps(json_data, ensure_ascii=False))


class PrimaryQuestionFindHandler(tornado.web.RequestHandler):
    def __init__(self):
        self.config = Config()
        self.search = Search()
        with open('../../data/primary_question_dict.json') as primary_dict_f:
            self.primary_question_dict = json.loads(primary_dict_f.readlines())

    def data_received(self, chunk):
        pass

    def get(self, *args, **kwargs):
        self.render('find_primary.html')

    def post(self, *args, **kwargs):
        self.use_write()

    def use_write(self):
        question = [self.get_argument('question')]
        try:
            results = self.search.search_by_question(question, top_n=5, config=self.config)
            question_list = [question for _ in range(len(results))]
            retrievaled_question = [temp[1] for temp in results]
            probs, _ = infer_prob(question_list, retrievaled_question, vocab_processor, model, session)
            positive_probs = probs[:, 1]
            max_probs_index = np.argmax(positive_probs)
            primary_question_id = results[max_probs_index, 2]
            if int(primary_question_id) == 0:
                return results[max_probs_index, 1]
            else:
                return self.primary_question_dict[results[max_probs_index, 0]]

        except Exception as e:
            print(e)


def make_app():
    setting = dict(
        template_path=os.path.join(os.path.dirname(__file__), 'templates'),
        static_path=os.path.join(os.path.dirname(__file__), 'static')
    )
    return tornado.web.Application([(r'/Match', MatchHandler)],
                                   **setting)


if __name__ == '__main__':
    app = make_app()
    app.listen(9003)
    tornado.ioloop.IOLoop.current().start()
