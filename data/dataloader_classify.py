#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : dataloader_classify.py
@Time     : 18-11-29 下午8:23
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import numpy as np
import tensorflow.contrib as tc
import pickle as pkl


divided_set = './question_classification.pkl'


def get_question_label(file_path):
    """
    data loader for question classification task
    :param file_path:
    :return:
    """
    questions = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                print('loading question classification data set successfully!')
                return [questions, np.array(labels)]
            tmp = line.strip().split('\t')
            question, label = tmp[0], tmp[1]
            questions.append(question)
            labels.append(label)


def batch_iter_per_ep4classify(questions, labels, batch_size=64, shuffle=True):

    data_len = len(labels)
    num_batch = int((data_len - 1) / batch_size) + 1

    if shuffle:
        indices = np.random.permutation(np.arange(data_len))
        questions_shuffled = questions[indices]
        labels_shuffled = labels[indices]
    else:
        questions_shuffled = questions
        labels_shuffled = labels
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield questions_shuffled[start_id: end_id], labels_shuffled[start_id: end_id]


def split_data(data_file, vocab_path, out_path, dev_sample_percentage=0.1, test_sample_percentage=0.1):
    questions, labels = get_question_label(data_file)

    # Build vocabulary
    vocab_processor = tc.learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    # padding to max length
    questions_pad = np.array(list(vocab_processor.transform(questions)))

    # Randomly shuffle data
    np.random.seed(7)
    shuffle_indices = np.random.permutation(np.arange(len(labels)))
    questions_shuffled = questions_pad[shuffle_indices]
    labels_shuffled = labels[shuffle_indices]

    # Split train/dev/test
    dev_sample_indices = -1 * int(dev_sample_percentage * float(len(labels)))
    test_sample_indices = -1 * int(test_sample_percentage * float(len(labels)))

    questions_train = questions_shuffled[: dev_sample_indices + test_sample_indices]
    questions_dev = questions_shuffled[dev_sample_indices + test_sample_indices : test_sample_indices]
    questions_test = questions_shuffled[test_sample_indices:]

    labels_train = labels_shuffled[: dev_sample_indices + test_sample_indices]
    labels_dev = labels_shuffled[dev_sample_indices + test_sample_indices: test_sample_indices]
    labels_test = labels_shuffled[test_sample_indices:]

    del questions, labels, questions_pad, questions_shuffled, labels_shuffled

    vocab_size = len(vocab_processor.vocabulary_)

    print("Vocabulary Size: {:d}".format(vocab_size))
    print("Train/Dev/test split: {:d}/{:d}/{:d}".format(len(labels_train), len(labels_dev), len(labels_test)))

    with open(out_path, 'wb') as pkl_file:
        try:
            pkl.dump(questions_train, pkl_file)
            pkl.dump(questions_dev, pkl_file)
            pkl.dump(questions_test, pkl_file)
            pkl.dump(labels_train, pkl_file)
            pkl.dump(labels_dev, pkl_file)
            pkl.dump(labels_test, pkl_file)
            pkl.dump(vocab_size, pkl_file)
        except Exception as e:
            print(e)


def load_pkl_set(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        question_train = pkl.load(pkl_file)
        question_dev = pkl.load(pkl_file)
        question_test = pkl.load(pkl_file)
        labels_train = pkl.load(pkl_file)
        labels_dev = pkl.load(pkl_file)
        labels_test = pkl.load(pkl_file)
        vocab_size = pkl.load(pkl_file)

    return question_train, question_dev, question_test, labels_train, labels_dev, labels_test, vocab_size


if __name__ == '__main__':
    pass
