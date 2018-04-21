#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
"""
import platform as plat
import os
import time

from general_function.file_wav import *
from general_function.file_dict import *
from general_function.gen_func import *

# LSTM_CNN
import keras as kr
import numpy as np
import random

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Reshape # , Flatten,LSTM,Convolution1D,MaxPooling1D,Merge
from keras.layers import Conv1D,LSTM,MaxPooling1D, Lambda, TimeDistributed, Activation,Conv2D, MaxPooling2D #, Merge,Conv1D
from keras import backend as K
from keras.optimizers import SGD, Adadelta

from readdata23 import DataSpeech
#from neural_network.ctc_layer import ctc_layer
#from neural_network.ctc_loss import ctc_batch_loss

#from keras.backend.tensorflow_backend import ctc_batch_cost

class ModelSpeech(): # 语音模型类
	def __init__(self, datapath):
		'''
		初始化
		默认输出的拼音的表示大小是1283，即1282个拼音+1个空白块
		'''
		MS_OUTPUT_SIZE = 1417
		self.MS_OUTPUT_SIZE = MS_OUTPUT_SIZE # 神经网络最终输出的每一个字符向量维度的大小
		#self.BATCH_SIZE = BATCH_SIZE # 一次训练的batch
		self.label_max_string_length = 64
		self.AUDIO_LENGTH = 1600
		self.AUDIO_FEATURE_LENGTH = 39
		self._model, self.base_model = self.CreateModel() 
		
		self.datapath = datapath
		self.slash = ''
		if(system_type == 'Windows'):
			self.slash='\\' # 反斜杠
		elif(system_type == 'Linux'):
			self.slash='/' # 正斜杠
		else:
			print('*[Message] Unknown System\n')
			self.slash='/' # 正斜杠
		if(self.slash != self.datapath[-1]): # 在目录路径末尾增加斜杠
			self.datapath = self.datapath + self.slash
	
		
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
		
		'''
		# 每一帧使用13维mfcc特征及其13维一阶差分和13维二阶差分表示，最大信号序列长度为1500
		input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))
		
		layer_h1 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(input_data) # 卷积层
		layer_h2 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1) # 卷积层
		layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2) # 池化层
		#layer_h3 = Dropout(0.2)(layer_h2) # 随机中断部分神经网络连接，防止过拟合
		layer_h4 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3) # 卷积层
		layer_h5 = Conv2D(64, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4) # 卷积层
		layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5) # 池化层
		
		#test=Model(inputs = input_data, outputs = layer_h6)
		#test.summary()
		
		layer_h7 = Reshape((400, 576))(layer_h6) #Reshape层
		#layer_h5 = LSTM(256, activation='relu', use_bias=True, return_sequences=True)(layer_h4) # LSTM层
		#layer_h6 = Dropout(0.2)(layer_h5) # 随机中断部分神经网络连接，防止过拟合
		layer_h8 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h7) # 全连接层
		layer_h9 = Dense(1417, use_bias=True, kernel_initializer='he_normal')(layer_h8) # 全连接层
		
		y_pred = Activation('softmax', name='Activation0')(layer_h9)
		model_data = Model(inputs = input_data, outputs = y_pred)
		#model_data.summary()
		
		
		#labels = Input(name='the_labels', shape=[60], dtype='float32')
		
		labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
		input_length = Input(name='input_length', shape=[1], dtype='int64')
		label_length = Input(name='label_length', shape=[1], dtype='int64')
		# Keras doesn't currently support loss funcs with extra parameters
		# so CTC loss is implemented in a lambda layer
		
		#layer_out = Lambda(ctc_lambda_func,output_shape=(self.MS_OUTPUT_SIZE, ), name='ctc')([y_pred, labels, input_length, label_length])#(layer_h6) # CTC
		loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
		
		
		
		model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
		
		model.summary()
		
		# clipnorm seems to speeds up convergence
		#sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
		ada_d = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
		
		#model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=["accuracy"])
		model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = ada_d)
		
		
		# captures output of softmax so we can decode the output during visualization
		test_func = K.function([input_data], [y_pred])
		
		print('[*提示] 创建模型成功，模型编译成功')
		return model, model_data
		
	def ctc_lambda_func(self, args):
		y_pred, labels, input_length, label_length = args
		
		y_pred = y_pred[:, :, :]
		#y_pred = y_pred[:, 2:, :]
		return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
	
	
	
	def TrainModel(self, datapath, epoch = 2, save_step = 1000, batch_size = 32, filename = 'model_speech/speech_model23'):
		'''
		训练模型
		参数：
			datapath: 数据保存的路径
			epoch: 迭代轮数
			save_step: 每多少步保存一次模型
			filename: 默认保存文件名，不含文件后缀名
		'''
		data=DataSpeech(datapath, 'train')
		#data.LoadDataList()
		num_data = data.GetDataNum() # 获取数据的数量
		for epoch in range(epoch): # 迭代轮数
			print('[running] train epoch %d .' % epoch)
			n_step = 0 # 迭代数据数
			while True:
				try:
					print('[message] epoch %d . Have train datas %d+'%(epoch, n_step*save_step))
					# data_genetator是一个生成器函数
					yielddatas = data.data_genetator(batch_size, self.AUDIO_LENGTH)
					#self._model.fit_generator(yielddatas, save_step, nb_worker=2)
					self._model.fit_generator(yielddatas, save_step)
					n_step += 1
				except StopIteration:
					print('[error] generator error. please check data format.')
					break
				
				self.SaveModel(comment='_e_'+str(epoch)+'_step_'+str(n_step * save_step))
				self.TestModel(self.datapath, str_dataset='train', data_count = 4)
				self.TestModel(self.datapath, str_dataset='dev', data_count = 4)
				
	def LoadModel(self,filename='model_speech/speech_model23.model'):
		'''
		加载模型参数
		'''
		self._model.load_weights(filename)
		self.base_model.load_weights(filename + '.base')

	def SaveModel(self,filename='model_speech/speech_model23',comment=''):
		'''
		保存模型参数
		'''
		self._model.save_weights(filename+comment+'.model')
		self.base_model.save_weights(filename + comment + '.model.base')

	def TestModel(self, datapath='', str_dataset='dev', data_count = 32, out_report = False):
		'''
		测试检验模型效果
		'''
		data=DataSpeech(self.datapath, str_dataset)
		#data.LoadDataList(str_dataset) 
		num_data = data.GetDataNum() # 获取数据的数量
		if(data_count <= 0 or data_count > num_data): # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
			data_count = num_data
		
		try:
			ran_num = random.randint(0,num_data - 1) # 获取一个随机数
			
			words_num = 0
			word_error_num = 0
			
			nowtime = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
			if(out_report == True):
				txt_obj = open('Test_Report_' + str_dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8') # 打开文件并读入
			
			txt = ''
			for i in range(data_count):
				data_input, data_labels = data.GetData((ran_num + i) % num_data)  # 从随机数开始连续向后取一定数量数据
				pre = self.Predict(data_input, data_input.shape[0] // 4)
				
				words_n = data_labels.shape[0] # 获取每个句子的字数
				words_num += words_n # 把句子的总字数加上
				edit_distance = GetEditDistance(data_labels, pre) # 获取编辑距离
				if(edit_distance <= words_n): # 当编辑距离小于等于句子字数时
					word_error_num += edit_distance # 使用编辑距离作为错误字数
				else: # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
					word_error_num += words_n # 就直接加句子本来的总字数就好了
				
				if(out_report == True):
					txt += str(i) + '\n'
					txt += 'True:\t' + str(data_labels) + '\n'
					txt += 'Pred:\t' + str(pre) + '\n'
					txt += '\n'
				
			
			print('*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率：', word_error_num / words_num * 100, '%')
			if(out_report == True):
				txt += '*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率： ' + str(word_error_num / words_num * 100) + ' %'
				txt_obj.write(txt)
				txt_obj.close()
			
		except StopIteration:
			print('[Error] Model Test Error. please check data format.')
	
	def Predict(self, data_input, input_len):
		'''
		预测结果
		返回语音识别后的拼音符号列表
		'''
		batch_size = 1 
		in_len = np.zeros((batch_size),dtype = np.int32)
		#print(in_len.shape)
		in_len[0] = input_len
		
		
		
		x_in = np.zeros((batch_size, 1600, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)
		
		for i in range(batch_size):
			x_in[i,0:len(data_input)] = data_input
		
		base_pred = self.base_model.predict(x = x_in)
		
		#print('base_pred:\n', base_pred)
		
		#y_p = base_pred
		#for j in range(200):
		#	mean = np.sum(y_p[0][j]) / y_p[0][j].shape[0]
		#	print('max y_p:',np.max(y_p[0][j]),'min y_p:',np.min(y_p[0][j]),'mean y_p:',mean,'mid y_p:',y_p[0][j][100])
		#	print('argmin:',np.argmin(y_p[0][j]),'argmax:',np.argmax(y_p[0][j]))
		#	count=0
		#	for i in range(y_p[0][j].shape[0]):
		#		if(y_p[0][j][i] < mean):
		#			count += 1
		#	print('count:',count)
		
		base_pred =base_pred[:, :, :]
		#base_pred =base_pred[:, 2:, :]
		r = K.ctc_decode(base_pred, in_len, greedy = True, beam_width=100, top_paths=1)
		#print('r', r)
		
		
		r1 = K.get_value(r[0][0])
		#print('r1', r1)
		
		#print('r0', r[1])
		r2 = K.get_value(r[1])
		#print(r2)
		#print('解码完成')
		r1=r1[0]
		
		return r1
		pass
	
	def RecognizeSpeech(self, wavsignal, fs):
		'''
		最终做语音识别用的函数，识别一个wav序列的语音
		不过这里现在还有bug
		'''
		
		#data = self.data
		#data = DataSpeech('E:\\语音数据集')
		#data.LoadDataList('dev')
		# 获取输入特征
		#data_input = GetMfccFeature(wavsignal, fs)
		#t0=time.time()
		data_input = GetFrequencyFeature(wavsignal, fs)
		#t1=time.time()
		#print('time cost:',t1-t0)
		
		input_length = len(data_input)
		input_length = input_length // 4
		
		data_input = np.array(data_input, dtype = np.float)
		#print(data_input,data_input.shape)
		data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
		
		r1 = self.Predict(data_input, input_length)
		
		list_symbol_dic = GetSymbolList(self.datapath) # 获取拼音列表
		
		
		r_str=[]
		for i in r1:
			r_str.append(list_symbol_dic[i])
		
		return r_str
		pass
		
	def RecognizeSpeech_FromFile(self, filename):
		'''
		最终做语音识别用的函数，识别指定文件名的语音
		'''
		
		wavsignal,fs = read_wav_data(filename)
		
		r = self.RecognizeSpeech(wavsignal, fs)
		
		return r
		
		pass
		
	
		
	@property
	def model(self):
		'''
		返回keras model
		'''
		return self._model


if(__name__=='__main__'):
	
	datapath = ''
	modelpath = 'model_speech'
	
	
	if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
		os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉
	
	system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
	if(system_type == 'Windows'):
		datapath = 'E:\\语音数据集'
		modelpath = modelpath + '\\'
	elif(system_type == 'Linux'):
		datapath = 'dataset'
		modelpath = modelpath + '/'
	else:
		print('*[Message] Unknown System\n')
		datapath = 'dataset'
		modelpath = modelpath + '/'
	
	ms = ModelSpeech(datapath)
	
	#ms.LoadModel(modelpath + 'speech_model23_e_0_step_1.model')
	ms.TrainModel(datapath, epoch = 50, batch_size = 4, save_step = 500)
	#ms.TestModel(datapath, str_dataset='train', data_count = 32, out_report = True)
	#r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\wav\\train\\A11\\A11_167.WAV')
	#r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\wav\\test\\D4\\D4_750.wav')
	#print('*[提示] 语音识别结果：\n',r)