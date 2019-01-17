#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : web_server9002.py
@Time     : 19-1-14 下午12:43
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
from ir.search4tk import Search4tk
from ir.config4tk import Config4tk
from logger_config import base_logger

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

from infer_tune import prepare, inference, infer_prob


def chinese_tokenizer(documents):
    """
    中文文本转换为词序列(restore时还需要用到，必须包含)
    :param documents:
    :return:
    """
    for document in documents:
        yield list(jieba.cut(document))


config = Config4tk()
search = Search4tk()
vocab_processor, model, session = prepare()
with open('../data/primary_question_dict.json') as primary_dict_f:
    primary_question_dict = json.loads(primary_dict_f.readline())


class FindAnswerHandler(tornado.web.RequestHandler):

    def data_received(self, chunk):
        pass

    def get(self, *args, **kwargs):
        self.render('find_answer.html')

    def post(self, *args, **kwargs):
        self.use_write()

    def use_write(self):
        question = self.get_argument('question')
        top_n = int(self.get_argument('top_n'))
        original = self.get_argument('original')
        if original == "weibao":
            try:
                results = search.search_by_question(question, top_n=top_n, config=config)
                question_list = [question for _ in range(len(results))]
                retrievaled_questions = [temp[1] for temp in results]
                probs, _ = infer_prob(question_list, retrievaled_questions, vocab_processor, model, session)
                positive_probs = probs[:, 1]
                alternative_answers = [temp[2] for temp in results]

                json_data = {'question': str(question),
                             'retrieval_question': '|||'.join(retrievaled_questions),
                             'probabilities': '|||'.join([str(prob) for prob in positive_probs]),
                             'answer': '|||'.join(alternative_answers)}
                base_logger.info(str(json_data))
                self.write(json.dumps(json_data, ensure_ascii=False))
            except Exception as e:
                base_logger.warning(str(e))
                json_data = {'question': str(question),
                             'retrieval_question': 'unknown',
                             'probabilities': 'unknown',
                             'answer': 'unknown'}
                self.write(json.dumps(json_data, ensure_ascii=False))
        else:
            self.write_error(500)


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
            # primary_question_ids = [temp[3] for temp in results]
            alternative_primary_questions = [temp[3] for temp in results]
            # for primary_question_id, result in zip(primary_question_ids, results):
            #     if int(primary_question_id) == 0:
            #         alternative_primary_questions.append(result[1])
            #     else:
            #         alternative_primary_questions.append(primary_question_dict[str(result[2])])
            json_data = {'alternative': '|||'.join(alternative_primary_questions),
                         'sub_question': '|||'.join(retrievaled_questions),
                         'match_score': '|||'.join([str(prob) for prob in positive_probs]),
                         'user_query': str(question)}
            base_logger.info(str(json_data))
            self.write(json.dumps(json_data, ensure_ascii=False))

        except Exception as e:
            # print(e)
            base_logger.warning(str(e))
            json_data = {'alternative': 'Unknown',
                         'match_score': '0',
                         'user_query': str(question)}
            self.write(json.dumps(json_data, ensure_ascii=False))


def make_app():
    setting = dict(
        template_path=os.path.join(os.path.dirname(__file__), 'templates'),
        static_path=os.path.join(os.path.dirname(__file__), 'static')
    )
    return tornado.web.Application([(r'/FindAnswer', FindAnswerHandler),
                                    (r'/FindPrimary', PrimaryQuestionFindHandler)],
                                   **setting)


if __name__ == '__main__':
    app = make_app()
    app.listen(9002)
    tornado.ioloop.IOLoop.current().start()
