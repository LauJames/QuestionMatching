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

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

from infer import prepare, inference


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
