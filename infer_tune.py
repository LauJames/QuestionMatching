#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : infer_tune.py
@Time     : 18-12-15 下午3:54
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import os
import sys
import time
import datetime
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import logging
import jieba
import pickle
from models.MVLSTM import MVLSTM

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))


def parse_args():
    parser = argparse.ArgumentParser('Question to Question matching for QA task')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--learning_rate', type=float, default=0.001, help='optimizer type')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['MVLSTM'], default='MVLSTM',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embedding_dim', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_q_len', type=int, default=30,
                                help='max length of question')
    model_settings.add_argument('--num_classes', type=int, default=2,
                                help='num of classes')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--tensorboard_dir', default='tensorboard_dir/MVLSTM_tune',
                               help='saving path of tensorboard')
    path_settings.add_argument('--save_dir', default='checkpoints/MVLSTM_tune',
                               help='save base dir')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')

    misc_setting = parser.add_argument_group('misc settings')
    misc_setting.add_argument('--allow_soft_placement', type=bool, default=True,
                              help='allow device soft device placement')
    misc_setting.add_argument('--log_device_placement', type=bool, default=False,
                              help='log placement of ops on devices')

    return parser.parse_args()


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def chinese_tokenizer(documents):
    """
    中文文本转换为词序列(restore时还需要用到，必须包含)
    :param documents:
    :return:
    """
    for document in documents:
        yield list(jieba.cut(document))


def prepare():
    args = parse_args()
    start_time = time.time()
    # absolute path
    save_path = os.path.join(curdir, os.path.join(args.save_dir, 'best_validation'))

    vocab_path = os.path.join(curdir, os.path.join(args.save_dir, 'vocab'))
    vocab_processor = tc.learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    model = MVLSTM(
        sequence_length=args.max_q_len,
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim,
        vocab_size=len(vocab_processor.vocabulary_),
        max_length=args.max_q_len,
        hidden_dim=args.hidden_size,
        learning_rate=args.learning_rate
    )

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, save_path=save_path)

    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)

    return vocab_processor, model, session


def inference(q1, q2, vocab_processor, model, session):
    # args = parse_args()
    # vocab_path = os.path.join(curdir, os.path.join(args.save_dir, 'vocab'))
    # vocab_processor = tc.learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    q1_pad = np.array(list(vocab_processor.transform(q1)))
    q2_pad = np.array(list(vocab_processor.transform(q2)))

    prediction = session.run(model.y_pred,
                             feed_dict={
                                 model.input_q1: q1_pad,
                                 model.input_q2: q2_pad,
                                 model.dropout_keep_prob: 1.0
                             })
    return prediction


def infer_prob(q1, q2, vocab_processor, model, session):
    q1_pad = np.array(list(vocab_processor.transform(q1)))
    q2_pad = np.array(list(vocab_processor.transform(q2)))

    predict_prob, prediction = session.run([model.probs, model.y_pred],
                                           feed_dict={
                                               model.input_q1: q1_pad,
                                               model.input_q2: q2_pad,
                                               model.dropout_keep_prob: 1.0
                                           })
    square_probs = predict_prob ** 2
    row_sum = np.sum(square_probs, axis=1)
    row_sum_duplicate = np.tile(np.reshape(row_sum, [-1, 1]), 2)
    aug_probs = square_probs / row_sum_duplicate  # a**2/(a**2 + b**2)    b**2/(a**2 + b**2)
    return aug_probs, prediction


if __name__ == '__main__':
    q1 = ['如何买保险', '如何买保险', '如何买保险']
    q2 = ['如何买保险', '你好，这个保险怎么买', '保险怎么买呢？']
    vocab_processor, model, session = prepare()
    probs, predict = infer_prob(q1, q2, vocab_processor, model, session)
    print(probs)
    print(predict)
    # q1 = ['如何买保险']
    # q2 = ['保险怎么买']
    # vocab_process, model, session = prepare()
    # prediction = inference(q1, q2, vocab_process, model, session)
    # print(prediction)
    # a = [0, 1, 2]
    # b = np.tile(np.reshape(a, [3, 1]), 2)
    # print(b)
