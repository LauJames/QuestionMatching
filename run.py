#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : run.py
@Time     : 2018/11/15 22:04
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
import csv
import logging
from sklearn import metrics
from models.MVLSTM import MVLSTM
from data.dataloader import load_data, batch_iter_per_epoch, get_q2q_label


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
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1, help='dropout keep prob')
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
    model_settings.add_argument('--num_classes', type=int, default=2,
                                help='num of classes')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['./data/q2q_pair.txt'],
                               help='list of files that contain the preprocessed data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['./data/q2q_pair_test.txt'])
    path_settings.add_argument('--tensorboard_dir', default='tensorboard_dir/MVLSTM',
                               help='saving path of tensorboard')
    path_settings.add_argument('--save_dir', default='checkpoints/MVLSTM',
                               help='save base dir')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')

    misc_setting = parser.add_argument_group('misc settings')
    misc_setting.add_argument('allow_soft_placement', type=bool, default=True,
                              help='allow device soft device placement')
    misc_setting.add_argument('log_device_placement', type=bool, default=False,
                              help='log placement of ops on devices')

    return parser.parse_args()


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def feed_data(q1_batch, q2_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_q1: q1_batch,
        model.input_q2: q2_batch,
        model.input_y: y_batch,
        model.dropout_keep_prob: keep_prob
    }
    return feed_dict


def evaluate(q1_dev, q2_dev, y_dev, sess):
    """
    Evaluate model on a dev set
    :param q1_dev:
    :param q2_dev:
    :param y_dev:
    :param sess:
    :return:
    """
    data_len = len(y_dev)
    batch_eval = batch_iter_per_epoch(q1_dev, q2_dev, y_dev)
    total_loss = 0.0
    total_acc = 0.0
    for q1_batch_eval, q2_batch_eval, y_batch_eval in batch_eval:
        batch_len = len(y_batch_eval)
        feed_dict = feed_data(q1_batch_eval, q2_batch_eval, y_batch_eval, keep_prob=1.0)
        loss, accuracy = sess.run([model.loss, model.accuracy], feed_dict)
        total_loss += loss * batch_len
        total_acc += accuracy * batch_len
    return total_loss/data_len, total_acc/data_len


def train():
    print('Configuring TensorBoard and Saver ...')
    tensorboard_dir = args.tensorboard_dir
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # Confirguring Saver

    saver = tf.train.Saver()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load data
    print('Loading data ...')
    start_time = time.time()
    q1_train, q2_train, y_train, q1_dev, q2_dev, y_dev = load_data(data_file=args.train_files,
                                                                   dev_sample_percentage=args.dev_sample_percentage,
                                                                   save_vocab_dir=args.save_dir,
                                                                   max_length=args.max_q_len)
    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)

    # Create Session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and Deviation ...')
    start_time = time.time()
    total_batch = 0
    best_acc_dev = 0.0
    last_improved = 0
    require_improvement = 30000  # Eearly stopping

    tag = False
    for epoch in range(args.epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter_per_epoch(q1_train, q2_train, y_train, args.batch_size)
        for q1_batch, q2_batch, y_batch in batch_train:
            feed_dict = feed_data(q1_train, q2_train, y_batch, args.dropout_keep_prob)
            if total_batch % args.checkpoint_every == 0:
                # write to tensorboard scalar
                summary = session.run(merged_summary, feed_dict)
                writer.add_summary(summary, total_batch)

            if total_batch % args.evaluate_every == 0:
                # print performance on train set and dev set
                feed_dict[model.dropout_keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_dev, acc_dev = evaluate(q1_dev, q2_dev, y_dev, session)

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


def test():
    print('Loading test data ...')
    start_time = time.time()
    q1_test, q2_test, y_test = get_q2q_label(args.test_data_files)

    vocab_path = os.path.join(args.save_dir, 'vocab')
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    q1 = np.array(list(vocab_processor.transform(q1_test)))
    q2 = np.array(list(vocab_processor.transform(q2_test)))

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, save_path=save_path)

    print('Testing ...')
    loss_test, acc_test = evaluate(q1_test, q2_test, y_test, session)
    print('Test loss:{0:>6.2}, Test acc:{1:7.2%}'.format(loss_test, acc_test))

    test_batches = batch_iter_per_epoch(q1, q2, y_test, shuffle=False)
    all_predictions = []
    all_predict_prob = []
    count = 0  # concatenate第一次不能为空，需要一个判断来赋all_predict_prob
    for q1_test_batch, q2_test_batch, y_test_batch in test_batches:
        batch_predictions, batch_predict_probs = session.run([model.y_pred, model.probs],
                                                             feed_dict={
                                                                 model.input_q1: q1_test_batch,
                                                                 model.input_q2: q2_test_batch,
                                                                 model.dropout_keep_prob: 1.0
                                                             })
        all_predictions = np.concatenate([all_predictions, batch_predictions])
        if count == 0:
            all_predict_prob = batch_predict_probs
        else:
            all_predict_prob = np.concatenate([all_predict_prob, batch_predict_probs])
        count = 1

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

    logger = logging.getLogger('q2q_matching')
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

    # MVLSTM model init
    model = MVLSTM(
        sequence_length=args.max_q_len,
        num_classes=args.num_classes,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_size,
        learning_rate=args.learning_rate
    )

    if args.prepare:
        pass
    if args.train:
        train()
    if args.evaluate:
        test()

