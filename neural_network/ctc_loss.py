#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.backend.tensorflow_backend import ctc_batch_cost
import tensorflow as tf

def ctc_batch_loss(y_true, y_pred):
	'''
		CTC的loss函数
		这里目前有bug
	'''
	loss = ctc_batch_cost(y_true, y_pred, tf.Variable((748,1283),dtype=tf.int64), tf.Variable((64,1283),dtype=tf.int64))
	return tf.Variable(loss,dtype=tf.int64)