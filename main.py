#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
"""
# LSTM_CNN
import keras as kr
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input # , Flatten,LSTM,Convolution1D,MaxPooling1D,Merge
from keras.layers import Conv1D,LSTM,MaxPooling1D, Lambda #, Merge, Conv2D, MaxPooling2D,Conv1D
from keras import backend as K

from readdata import DataSpeech

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
		定义CNN/LSTM/CTC模型，使用函数式模型
		输入层：39维的特征值序列，一条语音数据的最大长度设为1500（大约15s）
		隐藏层一：1024个神经元的卷积层
		隐藏层二：池化层，池化窗口大小为2
		隐藏层三：Dropout层，需要断开的神经元的比例为0.3，防止过拟合
		隐藏层四：循环层、LSTM层
		隐藏层五：Dropout层，需要断开的神经元的比例为0.3，防止过拟合
		输出层：全连接层，神经元数量为1279，使用softmax作为激活函数，使用CTC的loss作为损失函数
		'''
		# 每一帧使用13维mfcc特征及其13维一阶差分和13维二阶差分表示，最大信号序列长度为1500
		layer_input = Input((1500,39))
		
		layer_h1 = Conv1D(256, 5, use_bias=True, padding="valid")(layer_input) # 卷积层
		layer_h2 = MaxPooling1D(pool_size=2, strides=None, padding="valid")(layer_h1) # 池化层
		layer_h3 = Dropout(0.2)(layer_h2) # 随机中断部分神经网络连接，防止过拟合
		layer_h4 = LSTM(256, activation='relu', use_bias=True)(layer_h3) # LSTM层
		layer_h5 = Dropout(0.2)(layer_h4) # 随机中断部分神经网络连接，防止过拟合
		layer_h6 = Dense(1279, activation="softmax")(layer_h5) # 全连接层
		
		#labels = Input(name='the_labels', shape=[60], dtype='float32')
		layer_out = Lambda(ctc_lambda_func,output_shape=(1279,), name='ctc')(layer_h6) # CTC
		_model = Model(inputs = layer_input, outputs = layer_out)
		
		#_model = Sequential()
		
		#_model.add(Conv1D(256, 5,input_shape=(1500,39), use_bias=True, padding="valid"))
		#_model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))
		#_model.add(Dropout(0.3)) # 随机中断部分神经网络连接
		
		#_model.add(LSTM(256, activation='relu', use_bias=True))
		#_model.add(Dropout(0.3)) # 随机中断部分神经网络连接
		
		#_model.add(Dense(1279, activation="softmax"))
       ##_model.add(Lambda(ctc_lambda_func,output_shape=(1,),name='ctc'))
       
		#_model.compile(optimizer="sgd", loss='categorical_crossentropy',metrics=["accuracy"])
		_model.compile(optimizer="sgd", loss='ctc',metrics=["accuracy"])
		return _model
		
	def ctc_lambda_func(args):
		#labels, y_pred, input_length, label_length = args
		y_pred = args
		#y_pred = y_pred[:, 2:, :]
		return K.ctc_decode(y_pred,1279)
		#return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
	
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
			n_step = 0 # 迭代数据数
			while True:
				try:
					data_input, data_label = data.GetData(n_step) # 读数据
					
					pass
					# 需要写一个生成器函数
					self._model.fit_generator(yielddatas, save_step)
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

	def TestModel(self):
		'''
		测试检验模型效果
		'''
		pass

	def Predict(self,x):
		'''
		预测结果
		'''
		r = predict_on_batch(x)
		return r
		pass
		
	@property
	def model(self):
		'''
		返回keras model
		'''
		return self._model


if(__name__=='__main__'):
	pass
	
