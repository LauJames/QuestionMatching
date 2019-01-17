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
from typing import List, Any, Union

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

# from infer import prepare, inference, infer_prob
from infer_tune import prepare, inference, infer_prob


def chinese_tokenizer(documents):
    """
    中文文本转换为词序列(restore时还需要用到，必须包含)
    :param documents:
    :return:
    """
    for document in documents:
        yield list(jieba.cut(document))


config = Config()
search = Search()
vocab_processor, model, session = prepare()
with open('../data/primary_question_dict.json') as primary_dict_f:
    primary_question_dict = json.loads(primary_dict_f.readline())


class MatchHandler(tornado.web.RequestHandler):

    def data_received(self, chunk):
        pass

    def get(self, *args, **kwargs):
        self.render('find_answer.html')

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
    def data_received(self, chunk):
        pass

    def get(self, *args, **kwargs):
        self.render('find_primary.html')

    def post(self, *args, **kwargs):
        self.use_write()

    def use_write(self):
        question = self.get_argument('question')
        topn = int(self.get_argument('top_n'))
        try:
            results = search.search_by_question(question, top_n=topn, config=config)
            question_list = [question for _ in range(len(results))]
            retrievaled_questions = [temp[1] for temp in results]
            probs, _ = infer_prob(question_list, retrievaled_questions, vocab_processor, model, session)
            positive_probs = probs[:, 1]
            primary_question_ids = [temp[2] for temp in results]
            # positive_probs = probs[:, 1]
            # max_probs_index = np.argmax(positive_probs)
            # primary_question_id = results[max_probs_index][2]  # type: Union[List[Any], Any]
            alternative_primary_questions = []
            for primary_question_id, result in zip(primary_question_ids, results):
                if int(primary_question_id) == 0:
                    # json_data = {'primary_question': str(results[max_probs_index][1]),
                    #              'match_score': str(positive_probs[max_probs_index]),
                    #              'user_query': str(question)}
                    alternative_primary_questions.append(result[1])
                else:
                    alternative_primary_questions.append(primary_question_dict[str(result[2])])
                # json_data = {'primary_question': str(primary_question_dict[str(primary_question_id)]),
                #              'match_score': str(positive_probs[max_probs_index]),
                #              'user_query': str(question)}
                json_data = {'alternative': '|||'.join(alternative_primary_questions),
                             'sub_question': '|||'.join(retrievaled_questions),
                             'match_score': '|||'.join([str(prob) for prob in positive_probs]),
                             'user_query': str(question)}
            self.write(json.dumps(json_data, ensure_ascii=False))

        except Exception as e:
            print(e)
            json_data = {'alternative': 'Unknown',
                         'match_score': '0',
                         'user_query': str(question)}
            self.write(json.dumps(json_data, ensure_ascii=False))


def make_app():
    setting = dict(
        template_path=os.path.join(os.path.dirname(__file__), 'templates'),
        static_path=os.path.join(os.path.dirname(__file__), 'static')
    )
    return tornado.web.Application([(r'/Match', MatchHandler),
                                    (r'/FindPrimary', PrimaryQuestionFindHandler)],
                                   **setting)


if __name__ == '__main__':
    app = make_app()
    app.listen(9003)
    tornado.ioloop.IOLoop.current().start()
