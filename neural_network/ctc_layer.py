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
		对CTC层的实现，具体需要再去参考下论文...以及tensorflow中ctc的实现，
		并将其通过自定义层加入到keras的神经网络层中
	'''
	def __init__(self, input_dim, output_dim, **kwargs):
		super(ctc_layer, self).__init__(**kwargs)
		self.input_dim = input_dim
		self.output_dim = output_dim
		#self.input_spec = InputSpec(min_ndim=3)
		pass
	
	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=('''input_shape[0],''' self.output_dim, -1),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!
		
	
	def call(self, x, mask=None):
		decoded_dense, log_prob = K.ctc_decode(x,self.input_dim)
		decoded_sequence = K.ctc_label_dense_to_sparse(decoded_dense, decoded_dense.shape[0])
		return decoded_sequence
		
	
	def get_config(self):
		
		pass
	
	
	