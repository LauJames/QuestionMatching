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
from models.TextCNN import TextCNN
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
    model_settings.add_argument('--fiter_sizes', type=str, default='3,4,5',
                                help='Comma-separated filter sizes (default: "3,4,5")')
    model_settings.add_argument('num_filters', type=int, default=128,
                                help='Number of filters per filter size (default: 128)')
    model_settings.add_argument('--max_q_len', type=int, default=18,
                                help='max length of question')
    model_settings.add_argument('--l2_reg_lambda', type=float, default=0.0,
                                help='L2 regularization lambda (default: 0.0)')
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


def feed_data(question_batch, label_batch, keep_prob, model):
    feed_dict = {
        model.input_x: question_batch,
        model.input_y: label_batch,
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
    questions, labels = get_question_label(args.merged_files)
    start_time = time.time()
    vocab_processor = tc.learn.preprocessing.VocabularyProcessor(max_document_length=args.max_q_len,
                                                                 min_frequency=5,
                                                                 tokenizer_fn=chinese_tokenizer)
    questions_pad = np.array(list(vocab_processor.fit_transform(questions)))

    del questions, questions_pad, labels

    print('Vocab size: {}'.format(len(vocab_processor.vocabulary_)))
    vocab_processor.save(os.path.join(args.save_dir, "vocab"))

    # split
    split_data(args.merged_files, os.path.join(args.save_dir, "vocab"), args.pkl_files)

    time_dif = get_time_dif(start_time)
    print('Vocab processing time usage:', time_dif)


def train():
    # Load data
    print('Loading data ...')
    start_time = time.time()
    question_train, question_dev, question_test, labels_train, labels_dev, labels_test, vocab_size = load_pkl_set(args.pkl_files)

    del question_test, labels_test

    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)

    print('Configuring TensorBoard and Saver ...')
    tensorboard_dir = args.tensorboard_dir
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    # MVLSTM model init
    model = TextCNN(
        sequence_length=args.max_q_len,
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim,
        vocab_size=vocab_size,
        filter_sizes=list(map(int, args.filter_sizes.split(","))),
        num_filters=args.num_filters,
        l2_reg_lambda=args.l2_reg_lambda,
        learning_rate=args.learning_rate
    )

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # Configuring Saver

    saver = tf.train.Saver()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Create Session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and Deviation ...')
    start_time = time.time()
    total_batch = 0
    best_acc_dev = 0.0
    last_improved = 0
    require_improvement = 30000  # Early stopping

    tag = False
    for epoch in range(args.epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter_per_ep4classify(question_train, labels_train, args.batch_size)
        for question_batch, labels_batch in batch_train:
            feed_dict = feed_data(question_batch, labels_batch, args.dropout_keep_prob, model=model)
            if total_batch % args.checkpoint_every == 0:
                # write to tensorboard scalar
                summary = session.run(merged_summary, feed_dict)
                writer.add_summary(summary, total_batch)

            if total_batch % args.evaluate_every == 0:
                # print performance on train set and dev set
                feed_dict[model.dropout_keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_dev, acc_dev = evaluate(question_dev, labels_dev, session, model=model)

                if acc_dev > best_acc_dev:
                    # save best result
                    best_acc_dev = acc_dev
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                print('Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:7.2%}, Val Loss: {3:>6.2}, Val Acc:'
                      '{4:>7.2%}, Time:{5}{6}'
                      .format(total_batch, loss_train, acc_train, loss_dev, acc_dev, time_dif, improved_str))

            session.run(model.optim, feed_dict)
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # having no improvement for a long time
                print('No optimization for a long time, auto-stopping ...')
                tag = True
                break
        if tag:  # early stopping
            break


def predict():
    print('Loading test data ...')
    start_time = time.time()
    question_train, question_dev, question_test, labels_train, labels_dev, labels_test, vocab_size = load_pkl_set(
        args.pkl_files)

    del question_train, question_dev, labels_train, labels_dev

    # MVLSTM model init
    model = TextCNN(
        sequence_length=args.max_q_len,
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim,
        vocab_size=vocab_size,
        filter_sizes=list(map(int, args.filter_sizes.split(","))),
        num_filters=args.num_filters,
        l2_reg_lambda=args.l2_reg_lambda,
        learning_rate=args.learning_rate
    )

    # q1_pad = np.array(list(vocab_processor.transform(q1_test)))
    # q2_pad = np.array(list(vocab_processor.transform(q2_test)))

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, save_path=save_path)

    print('Testing ...')
    loss_test, acc_test = evaluate(question_test, labels_test, session, model=model)
    print('Test loss:{0:>6.2}, Test acc:{1:7.2%}'.format(loss_test, acc_test))

    test_batches = batch_iter_per_ep4classify(question_test, labels_test, shuffle=False)
    all_predictions = []
    all_predict_prob = []
    count = 0  # concatenate第一次不能为空，需要一个判断来赋all_predict_prob
    for question_batch, labels_batch in test_batches:
        batch_predictions, batch_predict_probs = session.run([model.y_pred, model.probs],
                                                             feed_dict={
                                                                 model.input_x: question_batch,
                                                                 model.dropout_keep_prob: 1.0
                                                             })
        all_predictions = np.concatenate([all_predictions, batch_predictions])
        if count == 0:
            all_predict_prob = batch_predict_probs
        else:
            all_predict_prob = np.concatenate([all_predict_prob, batch_predict_probs])
        count = 1

    y_test = [float(temp) for temp in labels_test]
    # Evaluation indices
    print('Precision, Recall, F1-Score ...')
    print(metrics.classification_report(y_test, all_predictions,
                                        target_names=['not match', 'match']))

    # Confusion Matrix
    print('Confusion Matrix ...')
    print(metrics.confusion_matrix(y_test, all_predictions))

    # Write probability to csv
    out_dir = os.path.join(args.save_dir, 'predict_prob_csv')
    print('Saving evaluation to {0}'.format(out_dir))
    with open(out_dir, 'w') as f:
        csv.writer(f).writerows(all_predict_prob)

    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)


if __name__ == '__main__':
    args = parse_args()

    save_path = os.path.join(args.save_dir, 'best_validation')

    logger = logging.getLogger('question-classification')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Runing with args: {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # if args.prepare:
    #     prepare()
    # if args.train:
    #     train()
    # if args.evaluate:
    #     predict()
    prepare()
    # train()
    # predict()