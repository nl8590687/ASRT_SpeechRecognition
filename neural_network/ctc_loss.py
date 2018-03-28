#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.backend.tensorflow_backend import ctc_batch_cost
import tensorflow as tf

def ctc_batch_loss(y_true, y_pred):
	'''
		CTC的loss函数
		这里目前有bug
	'''
	a=list()
	b=list()
	for i in range(0,32):
		a.append(748)
		b.append(64)
	
	#print(a,b)
	
	y_true_length = tf.Variable([1],dtype=tf.int64)
	y_pred_length = tf.Variable([1],dtype=tf.int64)
	
	#y_pred = y_pred[:, 2:, :]
	
	loss = ctc_batch_cost(y_true, y_pred, y_true_length, y_pred_length)
	return tf.Variable(loss,dtype=tf.int64)
	
def ctc_batch_loss2(y_true, y_pred):
	'''
		CTC的loss函数
		这里目前有bug
	'''
	#loss = ctc_batch_cost(y_true, y_pred, tf.Variable((748,1),dtype=tf.int64), tf.Variable((64,1),dtype=tf.int64))
	loss = tf.nn.ctc_loss(labels=y_true,inputs=y_pred, sequence_length=1500)
	return tf.Variable(loss,dtype=tf.int64)