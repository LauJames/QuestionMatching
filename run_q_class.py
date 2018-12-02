#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : run_q_class.py
@Time     : 18-12-2 下午3:36
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
import csv
import logging
import jieba
from sklearn import metrics
from models import TextCNN
from data.dataloader_classify import get_question_label, batch_iter_per_ep4classify, split_data, load_pkl_set


def parse_args():
    parser = argparse.ArgumentParser('Question to Question matching for QA task')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocab and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the match result fot test set on trained model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--dev_sample_percentage', type=float, default=0.1,
                                help='percentage of the training data to use for validation')
    train_settings.add_argument('--optim', default='adam', help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001, help='optimizer type')
    train_settings.add_argument('--weight_dacay', type=float, default=0, help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.5, help='dropout keep prob')
    train_settings.add_argument('--batch_size', type=int, default=64, help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10, help='train epochs')
    train_settings.add_argument('--evaluate_every', type=int, default=100,
                                help='evaluate model on dev set after this many training steps')
    train_settings.add_argument('--checkpoint_every', type=int, default=500,
                                help='save model after this many training steps')
    train_settings.add_argument('--num_checkpoints', type=int, default=5,
                                help='number of checkpoints to store')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['MVLSTM'], default='MVLSTM',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embedding_dim', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_q_len', type=int, default=18,
                                help='max length of question')
    model_settings.add_argument('--num_classes', type=int, default=1291,
                                help='num of classes')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files',
                               default='./data/question_label.txt',
                               # default='./data/test.txt',
                               help='list of files that contain the preprocessed data')
    path_settings.add_argument('--pkl_files',
                               default='./data/question_classification.pkl',
                               # default='./data/test.txt',
                               help='list of files that contain the preprocessed data')
    # path_settings.add_argument('--test_data_files',
    #                            default='./data/testset.txt')
    path_settings.add_argument('--tensorboard_dir', default='tensorboard_dir/CNN_Classification',
                               help='saving path of tensorboard')
    path_settings.add_argument('--save_dir', default='checkpoints/CNN_Classification',
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
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def feed_data(q1_batch, q2_batch, y_batch, keep_prob, model):
    feed_dict = {
        model.input_q1: q1_batch,
        model.input_q2: q2_batch,
        model.input_y: y_batch,
        model.dropout_keep_prob: keep_prob
    }
    return feed_dict


def evaluate(questions, labels, sess, model):
    """
    Evaluate model on a dev set
    :param q1_dev:
    :param q2_dev:
    :param y_dev:
    :param sess:
    :return:
    """
    data_len = len(labels)
    batch_eval = batch_iter_per_ep4classify(questions, labels)
    total_loss = 0.0
    total_acc = 0.0
    for question_batch_eval, label_batch_eval in batch_eval:
        batch_len = len(label_batch_eval)
        feed_dict = feed_data(question_batch_eval, label_batch_eval, keep_prob=1.0, model=model)
        loss, accuracy = sess.run([model.loss, model.accuracy], feed_dict)
        total_loss += loss * batch_len
        total_acc += accuracy * batch_len
    return total_loss/data_len, total_acc/data_len


def chinese_tokenizer(documents):
    """
    中文文本转换为词序列
    :param documents:
    :return:
    """
    for document in documents:
        yield list(jieba.cut(document))


def prepare():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print('Vocab processing ...')
    q1, q2, y = get_q2q_label(args.merged_files)
    start_time = time.time()
    vocab_processor = tc.learn.preprocessing.VocabularyProcessor(max_document_length=args.max_q_len,
                                                                 min_frequency=5,
                                                                 tokenizer_fn=chinese_tokenizer)
    q1_pad = np.array(list(vocab_processor.fit_transform(q1)))
    q2_pad = np.array(list(vocab_processor.fit_transform(q2)))

    del q1, q1_pad, q2, q2_pad, y

    print('Vocab size: {}'.format(len(vocab_processor.vocabulary_)))
    vocab_processor.save(os.path.join(args.save_dir, "vocab"))

    # split
    split_data(args.merged_files, os.path.join(args.save_dir, "vocab"), args.pkl_files)

    time_dif = get_time_dif(start_time)
    print('Vocab processing time usage:', time_dif)