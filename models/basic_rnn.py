#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : basic_rnn.py
@Time     : 2018/11/16 21:35
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""
import tensorflow as tf
import tensorflow.contrib as tc


def rnn(rnn_type, inputs, hidden_dim, num_layers=1, dropout_keep_prob=None, concat=True):
    """
    Implements (Bi-)LSTM, (Bi-)GRU and (Bi-)RNN
    :param rnn_type: the type of rnn
    :param inputs:  padded inputs
    :param length:  the valid length of the inputs
    :param hidden_dim:  the size of hidden units
    :param num_layers:  multiple rnn layer are stacked if layer_num > 1
    :param dropout_keep_prob:  the ratio of dropout
    :param concat:  When the rnn is bidirectional, the forward outputs and backward outputs
        are concated if this is True, else adding them
    :return:
        RNN outputs and final state
    """
    if not rnn_type.startswith('bi'):
        cell = get_cell(rnn_type, hidden_dim, num_layers, dropout_keep_prob)
        outputs, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        if rnn_type.endswith('lstm'):
            c = [state.c for state in states]
            h = [state.h for state in states]
            states = h
    else:
        cell_fw = get_cell(rnn_type, hidden_dim, num_layers, dropout_keep_prob)
        cell_bw = get_cell(rnn_type, hidden_dim, num_layers, dropout_keep_prob)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_bw, cell_fw, inputs, dtype=tf.float32
        )
        states_fw, states_bw = states
        if rnn_type.endswith('lstm'):
            c_fw = [state_fw.c for state_fw in states_fw]
            h_fw = [state_fw.h for state_fw in states_fw]
            c_bw = [state_bw.c for state_bw in states_bw]
            h_bw = [state_bw.h for state_bw in states_bw]
            states_fw, states_bw = h_fw, h_bw
        if concat:
            outputs = tf.concat(outputs, 2)
            states = tf.concat([states_fw, states_bw], 1)
        else:
            outputs = outputs[0] + outputs[1]
            states = states_fw + states_bw
    return outputs, states


def get_cell(rnn_type, hidden_dim, num_layers=1, dropout_keep_prob=None):
    """
    Gets the RNN cell
    :param rnn_type: ['lstm', 'gru', 'rnn']
    :param hidden_dim:  the size of hidden units
    :param num_layers:  MultiRNNCell are used if num_layers > 1
    :param dropout_keep_prob: the ratio of dropout
    :return:
        An RNN cell
    """
    def lstm_cell():
        return tc.rnn.LSTMCell(num_units=hidden_dim, state_is_tuple=True)

    def gru_cell():
        return tc.rnn.GRUCell(num_units=hidden_dim)

    def rnn_cell():
        return tc.rnn.BasicRNNCell(num_units=hidden_dim)

    def dropout():
        if rnn_type.endswith('lstm'):
            cell = lstm_cell()
        elif rnn_type.endswith('gru'):
            cell = gru_cell()
        elif rnn_type.endswith('rnn'):
            cell = rnn_cell()
        else:
            raise NotImplementedError('Unsupported rnn type: {}'.format(type))
        return tc.rnn.DropoutWrapper(cell,
                                     input_keep_prob=dropout_keep_prob,
                                     output_keep_prob=dropout_keep_prob)

    cells = [dropout() for _ in range(num_layers)]
    rnn_cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
    return rnn_cells
