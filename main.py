#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
"""
# LSTM_CNN
import keras as kr
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input # , Flatten,LSTM,Convolution1D,MaxPooling1D,Merge
from keras.layers import Conv1D,LSTM,MaxPooling1D, Lambda, TimeDistributed, Activation #, Merge, Conv2D, MaxPooling2D,Conv1D
from keras import backend as K

from readdata import DataSpeech
from neural_network.ctc_layer import ctc_layer
from neural_network.ctc_loss import ctc_batch_loss

class ModelSpeech(): # 语音模型类
	def __init__(self,MS_OUTPUT_SIZE = 1283,BATCH_SIZE = 32):
		'''
		初始化
		默认输出的拼音的表示大小是1283，即1282个拼音+1个空白块
		'''
		self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE # 神经网络最终输出的每一个字符向量维度的大小
		self.BATCH_SIZE = BATCH_SIZE # 一次训练的batch
		self._model = self.CreateModel() 

	def CreateModel(self):
		'''
		定义CNN/LSTM/CTC模型，使用函数式模型
		输入层：39维的特征值序列，一条语音数据的最大长度设为1500（大约15s）
		隐藏层一：1024个神经元的卷积层
		隐藏层二：池化层，池化窗口大小为2
		隐藏层三：Dropout层，需要断开的神经元的比例为0.2，防止过拟合
		隐藏层四：循环层、LSTM层
		隐藏层五：Dropout层，需要断开的神经元的比例为0.2，防止过拟合
		隐藏层六：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
		输出层：自定义层，即CTC层，使用CTC的loss作为损失函数，实现连接性时序多输出
		
		当前未完成，针对多输出的CTC层尚未实现
		'''
		# 每一帧使用13维mfcc特征及其13维一阶差分和13维二阶差分表示，最大信号序列长度为1500
		layer_input = Input((1500,39))
		
		layer_h1 = Conv1D(256, 5, use_bias=True, padding="valid")(layer_input) # 卷积层
		layer_h2 = MaxPooling1D(pool_size=2, strides=None, padding="valid")(layer_h1) # 池化层
		layer_h3 = Dropout(0.2)(layer_h2) # 随机中断部分神经网络连接，防止过拟合
		layer_h4 = LSTM(256, activation='relu', use_bias=True, return_sequences=True)(layer_h3) # LSTM层
		layer_h5 = Dropout(0.2)(layer_h4) # 随机中断部分神经网络连接，防止过拟合
		layer_h6 = Dense(self.MS_OUTPUT_SIZE, activation="softmax")(layer_h5) # 全连接层
		#layer_h6 = Dense(1283, activation="softmax")(layer_h5) # 全连接层
		
		layer_out = ctc_layer(self.MS_OUTPUT_SIZE, self.BATCH_SIZE)(layer_h6) # CTC层  可能有bug
		#layer_out = ctc_layer(1283, 32)(layer_h6) # CTC层  可能有bug
		
		#labels = Input(name='the_labels', shape=[60], dtype='float32')
		#layer_out = Lambda(ctc_lambda_func,output_shape=(self.MS_OUTPUT_SIZE, ), name='ctc')(layer_h6) # CTC
		#layer_out = TimeDistributed(Dense(self.MS_OUTPUT_SIZE, activation="softmax"))(layer_h5)
		_model = Model(inputs = layer_input, outputs = layer_out)
		
		
		#_model = Sequential()
		#_model.add(Conv1D(256, 5, use_bias=True, padding="valid", input_shape=(1500,39)))
		#_model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))
		#_model.add(Dropout(0.2))
		#_model.add(LSTM(256, activation='relu', use_bias=True, return_sequences=True))
		#_model.add(Dropout(0.2))
		#_model.add(TimeDistributed(Dense(self.MS_OUTPUT_SIZE)))
		#_model.add(Activation("softmax"))
		
		
		
		
		#_model.compile(optimizer="sgd", loss='categorical_crossentropy',metrics=["accuracy"])
		_model.compile(optimizer = "sgd", loss = ctc_batch_loss, metrics = ["accuracy"])
		return _model
		
	'''
	def ctc_lambda_func(args):
		#labels, y_pred, input_length, label_length = args
		y_pred = args[:,2:,:]
		#y_pred = y_pred[:, 2:, :]
		return K.ctc_decode(y_pred,self.MS_OUTPUT_SIZE)
		#return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
	'''
	
	def TrainModel(self,datapath,epoch = 2,save_step=1000,filename='model_speech/LSTM_CNN_model'):
		'''
		训练模型
		参数：
			datapath: 数据保存的路径
			epoch: 迭代轮数
			save_step: 每多少步保存一次模型
			filename: 默认保存文件名，不含文件后缀名
		'''
		data=DataSpeech(datapath)
		data.LoadDataList('train')
		num_data=DataSpeech.GetDataNum() # 获取数据的数量
		for epoch in range(epoch): # 迭代轮数
			print('[running] train epoch %d .' % epoch)
			n_step = 0 # 迭代数据数
			while True:
				try:
					print('[message] epoch %d . Have train datas %d+'%(epoch, n_step*save_step))
					# data_genetator是一个生成器函数
					yielddatas = data.data_genetator(self.BATCH_SIZE)
					self._model.fit_generator(yielddatas, save_step, nb_worker=2)
					n_step += 1
				except StopIteration:
					print('[error] generator error. please check data format.')
					break
				
				self.SaveModel(comment='_e_'+str(epoch)+'_step_'+str(n_step))
				
				
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

	def TestModel(self, datapath, str_dataset='dev'):
		'''
		测试检验模型效果
		'''
		data=DataSpeech(datapath)
		data.LoadDataList(str_dataset) 
		num_data = DataSpeech.GetDataNum() # 获取数据的数量
		try:
			gen = data.data_genetator(num_data)
			for i in range(1):
				X, y = gen
			r = self._model.test_on_batch(X, y)
			print(r)
		except StopIteration:
			print('[Error] Model Test Error. please check data format.')

	def Predict(self,x):
		'''
		预测结果
		'''
		r = self._model.predict_on_batch(x)
		print(r)
		return r
		pass
		
	@property
	def model(self):
		'''
		返回keras model
		'''
		return self._model


if(__name__=='__main__'):
	datapath = 'E:\\语音数据集'
	ms = ModelSpeech()
	#ms.TrainModel(datapath)
	#ms.TestModel(datapath)
	
