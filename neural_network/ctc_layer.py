#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
本代码用来实现神经网络中的CTC层
CTC层即：Connectionist Temporal Classification （连续型短时分类）
将这里实现的
尚未完成
'''

from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K

try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations

import tensorflow as tf

# 继承父类Layer
class ctc_layer(Layer):
	'''
		本类是对CTC层的实现，具体请去参考下论文...
		tensorflow中和keras中有ctc的一些实现，
		并将其通过自定义层加入到keras创建的神经网络层中
		
		参数：
			output_dim: 每一条时间序列中，输出的标签序列张量的尺寸长度
			
			
		目前可能有bug
	'''
	def __init__(self, output_dim, batch_size, **kwargs):
		'''
			这里是神经网络CTC层的初始化模块
		'''
		#if 'input_shape' not in kwargs and 'input_dim' in kwargs:
        #    kwargs['input_shape'] = (kwargs.pop('input_dim'), kwargs.pop('input_dim'),)
		super(ctc_layer, self).__init__(**kwargs)
		#self.input_dim = input_dim
		#self.input_spec = [InputSpec(dtype=(None,,output_dim),ndim=3, axes={None: input_dim})]
		self.output_dim = output_dim
		self.batch_size = batch_size
		#self.input_spec = InputSpec(min_ndim=3)
		#super(ctc_layer, self).build(input_shape)  # Be sure to call this somewhere!
		pass
	
	def build(self, input_shape):
		assert len(input_shape) >= 2
		#input_dim = input_shape[-1]
		# Create a trainable weight variable for this layer.
		self.kernel = self.add_weight(name='kernel', 
										shape=(input_shape[1], input_shape[2]), 
										initializer='uniform', 
										trainable=True)
		
		#super(ctc_layer, self).build(input_shape)  # Be sure to call this somewhere!
		#self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
		self.input_spec = [InputSpec(min_ndim=3)] # , axes={1: 748, -1: self.output_dim}
		#self.built = True
		self.built = False
	
	def call(self, x, mask=None):
		#output = K.dot(x, self.kernel)
		output = x
		
		#output.shape[0] = self.batch_size
		decoded_dense, log_prob = K.ctc_decode(output,tf.Variable((output.shape[1],output.shape[2]),dtype=tf.int64))
		#decoded_dense, log_prob = K.ctc_decode(output,output.shape[1])
		#decoded_sequence = K.ctc_label_dense_to_sparse(decoded_dense, len(decoded_dense))
		#return decoded_sequence
		return decoded_dense
	
	def get_config(self):
		config = {
			'output_dim': self.output_dim
		}
		base_config = super(ctc_layer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
	
	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim, input_shape[2])
	