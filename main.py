#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
"""
# LSTM_CNN
import keras as kr
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten # ,Input,LSTM,Convolution1D,MaxPooling1D,Merge
from keras.layers import Conv1D,LSTM,MaxPooling1D,Merge # Conv2D, MaxPooling2D,Conv1D

class ModelSpeech(): # 语音模型类
	def __init__(self,MS_EMBED_SIZE = 64,BATCH_SIZE = 32):
		'''
		初始化
		'''
		self.MS_EMBED_SIZE = MS_EMBED_SIZE # LSTM 的大小
		self.BATCH_SIZE = BATCH_SIZE # 一次训练的batch
		self._model = self.CreateModel() 

	def CreateModel(self):
		'''
		定义LSTM/CNN模型，尚未完成
		'''
		_model = Sequential()
		_model.add(LSTM(self.MS_EMBED_SIZE, return_sequences=True, input_shape = (200,400))) # input_shape需要修改
		_model.add(Dropout(0.3))
		_model.add(Conv1D(self.QA_EMBED_SIZE // 2, 5, border_mode="valid"))
		_model.add(MaxPooling1D(pool_length=2, border_mode="valid"))
		_model.add(Dropout(0.3))
		_model.add(Flatten())
		
		
		
		#_model = Sequential()
		#_model.add(Merge([m_lstm, aenc], mode="concat", concat_axis=-1))
		_model.add(Dense(1279, activation="softmax"))
		_model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=["accuracy"])
		return _model

	def TrainModel(self,datapath,epoch = 2,save_step=1000,filename='model_speech/LSTM_CNN_model'):
		'''
		训练模型
		参数：
			datapath: 数据保存的路径
			epoch: 迭代轮数
			save_step: 每多少步保存一次模型
			filename: 默认保存文件名，不含文件后缀名
		'''
		for epoch in range(epoch):
			pass
		pass

	def LoadModel(self,filename='model_speech/LSTM_CNN_model.model'):
		'''
		加载模型参数
		'''
		self._model.load_weights(filename)

	def SaveModel(self,filename='model_speech/LSTM_CNN_model',comment=''):
		'''
		保存模型参数
		'''
		self._model.save_weights(filename+comment+'.model')

	def TestModel(self):
		'''
		测试检验模型效果
		'''
		pass

	@property
	def model(self):
		'''
		返回keras model
		'''
		return self._model
	

if(__name__=='__main__'):
	pass
	
