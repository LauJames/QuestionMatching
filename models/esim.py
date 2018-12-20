#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : esim.py
@Time     : 18-12-17 下午3:29
@Software : PyCharm
@Reference: https://github.com/HsiaoYetGun/ESIM
"""
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
from models.basic_rnn import rnn


class ESIM(object):
    """
    ESIM model implemented by tensorflow, using for Question-Question matching task
    """
    def __init__(self, sequence_length, num_classes, embedding_dim, vocab_size,
                 max_length, hidden_dim, learning_rate, l2_lambda=0.0001, optimizer='adam'):

        # placeholders for input, label and dropout_prob
        self.input_q1 = tf.placeholder(tf.int64, [None, sequence_length], name='input_q1')
        self.input_q2 = tf.placeholder(tf.int64, [None, sequence_length], name='input_q2')
        self.input_y = tf.placeholder(tf.int64, [None], name='input_y')
        # self.q1_mask = tf.placeholder(tf.int64, [None], name='q1_mask')
        # self.q2_mask = tf.placeholder(tf.int64, [None], name='q2_mask')
        # self.input_y_onehot = tf.one_hot(self.input_y, 2, dtype=tf.int64)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('word_embedding'):
            # pre-trained word vector
            # self.word_embeddings = tf.get_variable(
            #     shape=(self.vocab.size(), embedding_dim),
            #     initializer=tf.constant_initializer(self.vocab.embeddings),
            #     trainable=True,
            #     name='word_embeddings'
            # )

            # random initially word vector
            self.word_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0),
                                               name='word_embeddings')
            self.q1_emb = tf.nn.embedding_lookup(self.word_embeddings, self.input_q1)
            self.q2_emb = tf.nn.embedding_lookup(self.word_embeddings, self.input_q2)

        # Encoding Layer
        # RNN model
        with tf.variable_scope('q1_encoding'):
            self.sep_q1_encodes, _ = rnn('bi-lstm', self.q1_emb, hidden_dim,
                                         # seq_len=self.q1_mask,
                                         dropout_keep_prob=self.dropout_keep_prob,
                                         concat=True)
        with tf.variable_scope('q2_encoding'):
            self.sep_q2_encodes, _ = rnn('bi-lstm', self.q2_emb, hidden_dim,
                                         # seq_len=self.q2_mask,
                                         dropout_keep_prob=self.dropout_keep_prob,
                                         concat=True)

        # local inference block
        with tf.variable_scope('local_inference'):
            self.attention_weights = tf.matmul(self.sep_q1_encodes, tf.transpose(self.sep_q2_encodes, [0, 2, 1]))
            # also can implemented by this way
            # attention_weights = tf.einsum('abd,acd->abc', self.sep_q1_encodes, self.sep_q2_encodes)

            self.attentionSoft_q1 = tf.nn.softmax(self.attention_weights)
            self.attentionSoft_q2 = tf.nn.softmax(tf.transpose(self.attention_weights))
            self.attentionSoft_q2 = tf.transpose(self.attentionSoft_q2)

            self.q1_hat = tf.matmul(self.attentionSoft_q1, self.sep_q2_encodes)
            self.q2_hat = tf.matmul(self.attentionSoft_q2, self.sep_q1_encodes)

            self.q1_diff = tf.subtract(self.sep_q1_encodes, self.q1_hat)
            self.q2_diff = tf.subtract(self.sep_q2_encodes, self.q2_hat)

            self.q1_mul = tf.multiply(self.sep_q1_encodes, self.q1_hat)
            self.q2_mul = tf.multiply(self.sep_q2_encodes, self.q2_hat)

            self.m_q1 = tf.concat([self.sep_q1_encodes, self.q1_hat, self.q1_diff, self.q1_mul], axis=2)
            self.m_q2 = tf.concat([self.sep_q2_encodes, self.q2_hat, self.q2_diff, self.q2_mul], axis=2)

        # composition block
        with tf.variable_scope('inference_composition'):
            with tf.variable_scope('composition_biLSTM1'):
                self.v_q1, _ = rnn('bi-lstm', self.m_q1, hidden_dim,
                                   dropout_keep_prob=self.dropout_keep_prob,
                                   concat=True)
            # with tf.variable_scope('composition_biLSTM', reuse=True):
            with tf.variable_scope('composition_biLSTM2'):
                self.v_q2, _ = rnn('bi-lstm', self.m_q2, hidden_dim,
                                   dropout_keep_prob=self.dropout_keep_prob,
                                   concat=True)
                self.v_q1_avg = tf.reduce_mean(self.v_q1, axis=1)
                self.v_q2_avg = tf.reduce_mean(self.v_q2, axis=1)
                self.v_q1_max = tf.reduce_max(self.v_q1, axis=1)
                self.v_q2_max = tf.reduce_max(self.v_q2, axis=1)

                self.v = tf.concat([self.v_q1_avg, self.v_q1_max, self.v_q2_avg, self.v_q2_max], axis=1)

        with tf.variable_scope('feed_forward'):
            with tf.variable_scope('feed_forward_layer1'):
                self.inputs1 = tf.nn.dropout(self.v, self.dropout_keep_prob)
                self.outputs1 = tf.layers.dense(self.inputs1, hidden_dim,
                                                activation=tf.nn.relu,
                                                kernel_initializer=tf.random_normal_initializer(0.0, 1.0))
            with tf.variable_scope('feed_forward_layer2'):
                self.outputs2 = tf.nn.dropout(self.outputs1, self.dropout_keep_prob)
                self.logits = tf.layers.dense(self.outputs2,
                                              units=num_classes,
                                              activation=tf.nn.tanh,
                                              kernel_initializer=tf.random_normal_initializer(0.0, 1.0))

        with tf.variable_scope('loss'):
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits + 1e-10)
            self.loss = tf.reduce_mean(self.losses, name='loss')
            # self.weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            # self.l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in self.weights]) * l2_lambda
            # self.loss += self.l2_loss

        with tf.variable_scope('train_op'):
            if optimizer == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            elif optimizer == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
            elif optimizer == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate).minimize(self.loss)
            elif optimizer == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            elif optimizer == 'adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(self.loss)
            elif optimizer == 'adagrad':
                self.optimizer = tf.train.AdagradDAOptimizer(learning_rate).minimize(self.loss)
            else:
                ValueError('Unknown optimizer: {0}'.format(optimizer))

        with tf.variable_scope('acc'):
            self.probs = tf.nn.softmax(self.logits, name='probability')
            self.predict = tf.argmax(self.probs, axis=1, name='predict_label')
            self.correct_predict = tf.equal(tf.cast(self.predict, tf.int64), self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predict, tf.float32), name='Accuracy')
