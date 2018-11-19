#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : MVLSTM.py
@Time     : 18-11-14 下午5:37
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import tensorflow as tf
import tensorflow.contrib as tc
from models.basic_rnn import rnn


class MVLSTM(object):
    """
    MV-LSTM implemented by tensorflow, using for Question-Question match
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_dim,
                 max_length, num_layers, hidden_dim, learning_rate, top_k=100):
        # Placeholders for input, output and dropout
        self.input_q1 = tf.placeholder(tf.int32, [None, sequence_length], name='input_q1')
        self.input_q2 = tf.placeholder(tf.int32, [None, sequence_length], name='input_q2')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True,
                name='word_embeddings'
            )
            self.q1_emb = tf.nn.embedding_lookup(self.word_embeddings, self.input_q1)
            self.q2_emb = tf.nn.embedding_lookup(self.word_embeddings, self.input_q2)

        # Encoding layer
        # RNN model
        with tf.variable_scope('q1_encoding'):
            self.sep_q1_encodes, _ = rnn('bi-lstm', self.q1_emb, max_length, hidden_dim)
        with tf.variable_scope('q2_encoding'):
            self.sep_q2_encodes, _ = rnn('bi-lstm', self.q2_emb, max_length, hidden_dim)

        # Match layer
        # dot-match
        with tf.name_scope('match'):
            self.cross = tf.einsum('abd,acd->abc', self.sep_q1_encodes, self.sep_q2_encodes)
            self.cross = tf.expand_dims(self.cross, 3)

        with tf.name_scope('score'):
            self.cross_reshape = tf.reshape(self.cross, (-1,), 'reshape')

            self.mm_k = tf.nn.top_k(self.cross_reshape, top_k, sorted=True)[0]

            self.pool1_flat_drop = tf.nn.dropout(self.mm_k, self.dropout_keep_prob)

            # Dense and classifier
            self.logits = tf.layers.dense(self.pool1_flat_drop, num_classes, name='logits')
            self.probs = tf.nn.softmax(self.logits)
            # prediction
            self.y_pred = tf.argmax(self.probs, 1)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # optimizer
            self.optim = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.y_pred, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
