#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
"""
import platform as plat
import os

from general_function.file_wav import *
import numpy as np

# LSTM_CNN
import keras as kr
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input # , Flatten,LSTM,Convolution1D,MaxPooling1D,Merge
from keras.layers import Conv1D,LSTM,MaxPooling1D, Lambda, TimeDistributed, Activation #, Merge, Conv2D, MaxPooling2D,Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate

from keras.layers.advanced_activations import LeakyReLU

from keras import backend as K
from keras.optimizers import SGD, Adadelta

from readdata import DataSpeech
from neural_network.ctc_layer import ctc_layer
from neural_network.ctc_loss import ctc_batch_loss

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
		self.AUDIO_FEATURE_LENGTH = 200
		self._model, self.base_model = self.CreateModel() 
		
		self.data = DataSpeech(datapath)
		
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
		输出层：自定义层，即CTC层，使用CTC的loss作为损失函数
		
		当前未完成，网络模型可能还需要修改
		'''
		# 每一帧使用13维mfcc特征及其13维一阶差分和13维二阶差分表示，最大信号序列长度为1500
		input_data = Input(name='the_input', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH))
		
		layer_h1_c = Conv1D(filters=256, kernel_size=5, strides=1, use_bias=True, kernel_initializer='he_normal', padding="same")(input_data) # 卷积层
		layer_h1_a = LeakyReLU(alpha=0.3)(layer_h1_c) # 高级激活层
		layer_h2_c = Conv1D(filters=256, kernel_size=5, strides=1, use_bias=True, kernel_initializer='he_normal', padding="same")(layer_h1_a) # 卷积层
		#layer_h1_a = Activation('relu', name='relu0')(layer_h1_c)
		layer_h2_a = LeakyReLU(alpha=0.3)(layer_h2_c) # 高级激活层
		layer_h3 = MaxPooling1D(pool_size=2, strides=None, padding="valid")(layer_h2_a) # 池化层
		
		layer_h4 = BatchNormalization()(layer_h3)
		
		layer_h4_c = Conv1D(filters=256, kernel_size=5, strides=1, use_bias=True, kernel_initializer='he_normal', padding="same")(layer_h4) # 卷积层
		layer_h4_a = LeakyReLU(alpha=0.3)(layer_h4_c) # 高级激活层
		layer_h5_c = Conv1D(filters=256, kernel_size=5, strides=1, use_bias=True, kernel_initializer='he_normal', padding="same")(layer_h4_a) # 卷积层
		layer_h5_a = LeakyReLU(alpha=0.3)(layer_h5_c) # 高级激活层
		#layer_h3_a = Activation('relu', name='relu1')(layer_h3_c)
		layer_h6 = MaxPooling1D(pool_size=2, strides=None, padding="valid")(layer_h5_a) # 池化层
		
		layer_h4 = Dropout(0.1)(layer_h3) # 随机中断部分神经网络连接，防止过拟合
		
		layer_h7 = Dense(256, use_bias=True, kernel_initializer='he_normal', activation="relu")(layer_h6) # 全连接层
		layer_h8 = Dense(256, use_bias=True, kernel_initializer='he_normal', activation="relu")(layer_h7) # 全连接层
		#layer_h4 = Activation('softmax', name='softmax0')(layer_h4_d1)
		
		layer_h8a = LSTM(256, activation='tanh', use_bias=True, return_sequences=True, kernel_initializer='he_normal')(layer_h8) # LSTM层
		layer_h8b = LSTM(256, activation='tanh', use_bias=True, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(layer_h8) # LSTM层
		
		layer_h8_merged = add([layer_h8a, layer_h8b])
		
		layer_h9a = LSTM(256, activation='tanh', use_bias=True, return_sequences=True, kernel_initializer='he_normal')(layer_h8_merged) # LSTM层
		layer_h9b = LSTM(256, activation='tanh', use_bias=True, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')(layer_h8_merged) # LSTM层
		
		layer_h9 = concatenate([layer_h9a, layer_h9b])
		#layer_h10 = Activation('softmax', name='softmax1')(layer_h9)
		
		#layer_h10_dropout = Dropout(0.1)(layer_h10) # 随机中断部分神经网络连接，防止过拟合
		
		#layer_h11 = Dense(512, use_bias=True, activation="softmax")(layer_h8) # 全连接层
		layer_h10 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h9) # 全连接层
		#layer_h6 = Dense(1283, activation="softmax")(layer_h5) # 全连接层
		
		y_pred = Activation('softmax', name='softmax2')(layer_h10)
		model_data = Model(inputs = input_data, outputs = y_pred)
		#self.base_model = model_data
		#model_data.summary()
		
		
		#labels = Input(name='the_labels', shape=[60], dtype='float32')
		
		labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
		input_length = Input(name='input_length', shape=[1], dtype='int64')
		label_length = Input(name='label_length', shape=[1], dtype='int64')
		# Keras doesn't currently support loss funcs with extra parameters
		# so CTC loss is implemented in a lambda layer
		
		#layer_out = Lambda(ctc_lambda_func,output_shape=(self.MS_OUTPUT_SIZE, ), name='ctc')([y_pred, labels, input_length, label_length])#(layer_h6) # CTC
		loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
		
		#y_out = Activation('softmax', name='softmax3')(loss_out)
		model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
		
		model.summary()
		
		# clipnorm seems to speeds up convergence
		#sgd = SGD(lr=0.0001, decay=1e-8, momentum=0.9, nesterov=True, clipnorm=5)
		ada_d = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
		
		#model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = sgd, metrics=['accuracy'])
		model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = ada_d, metrics=['accuracy'])
		
		
		# captures output of softmax so we can decode the output during visualization
		self.test_func = K.function([input_data], [y_pred])
		#top_k_decoded, _ = K.ctc_decode(y_pred, input_length, greedy = True, beam_width=100, top_paths=1)
		#self.decoder = K.function([input_data, input_length], [top_k_decoded[0]])
		
		print('[*提示] 创建模型成功，模型编译成功')
		return model, model_data
		
	def ctc_lambda_func(self, args):
		y_pred, labels, input_length, label_length = args
		#print(y_pred)
		y_pred = y_pred[:, 2:, :]
		#return K.ctc_decode(y_pred,self.MS_OUTPUT_SIZE)
		return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
	
	
	
	def TrainModel(self, datapath, epoch = 2, batch_size = 32, save_step = 1000, filename = 'model_speech/speech_model'):
		'''
		训练模型
		参数：
			datapath: 数据保存的路径
			epoch: 迭代轮数
			save_step: 每多少步保存一次模型
			filename: 默认保存文件名，不含文件后缀名
		'''
		#data=DataSpeech(datapath)
		data = self.data
		data.LoadDataList('train')
		num_data = data.GetDataNum() # 获取数据的数量
		for epoch in range(epoch): # 迭代轮数
			print('[running] train epoch %d .' % epoch)
			n_step = 0 # 迭代数据数
			while (n_step * save_step < num_data):
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
				
				
	def LoadModel(self, filename = 'model_speech/speech_model_e_0_step_1.model'):
		'''
		加载模型参数
		'''
		self._model.load_weights(filename)
		self.base_model.load_weights(filename + '.base')
		print('*[提示] 已加载模型')

	def SaveModel(self, filename = 'model_speech/speech_model', comment = ''):
		'''
		保存模型参数
		'''
		self._model.save_weights(filename + comment + '.model')
		self.base_model.save_weights(filename + comment + '.model.base')

	def TestModel(self, datapath, str_dataset='dev', data_count = 32):
		'''
		测试检验模型效果
		'''
		#data=DataSpeech(datapath)
		data = self.data
		data.LoadDataList(str_dataset) 
		num_data = data.GetDataNum() # 获取数据的数量
		if(data_count <= 0 or data_count > num_data): # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
			data_count = num_data
		
		try:
			gen = data.data_genetator(data_count)
			#for i in range(1):
			#	[X, y, input_length, label_length ], labels = gen
			#r = self._model.test_on_batch([X, y, input_length, label_length ], labels)
			r = self._model.evaluate_generator(generator = gen, steps = 1, max_queue_size = data_count, workers = 1, use_multiprocessing = False)
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
		
	def decode_batch(self, test_func, word_batch):
		out = test_func([word_batch])[0]
		ret = []
		for j in range(out.shape[0]):
			out_best = list(np.argmax(out[j, 2:], 1))
			out_best = [k for k, g in itertools.groupby(out_best)]
			outstr = labels_to_text(out_best)
			ret.append(outstr)
		return ret
	
	def show_edit_distance(self, num):
		num_left = num
		mean_norm_ed = 0.0
		mean_ed = 0.0
		while num_left > 0:
			word_batch = next(self.text_img_gen)[0]
			num_proc = min(word_batch['the_input'].shape[0], num_left)
			decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
			for j in range(num_proc):
				edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
				mean_ed += float(edit_dist)
				mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
			num_left -= num_proc
		mean_norm_ed = mean_norm_ed / num
		mean_ed = mean_ed / num
		print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
				% (num, mean_ed, mean_norm_ed))
	
	def RecognizeSpeech(self, wavsignal, fs):
		'''
		最终做语音识别用的函数，识别一个wav序列的语音
		不过这里现在还有bug
		'''
		#data = self.data
		data = DataSpeech('E:\\语音数据集')
		data.LoadDataList('dev')
		# 获取输入特征
		#data_input = data.GetMfccFeature(wavsignal, fs)
		data_input = data.GetFrequencyFeature(wavsignal, fs)
		input_length = len(data_input)
		input_length = input_length // 4
		
		data_input = np.array(data_input, dtype = np.float)
		in_len = np.zeros((1),dtype = np.int32)
		print(in_len.shape)
		in_len[0] = input_length -2
		
		
		batch_size = 1 
		x_in = np.zeros((batch_size, 1600, 200), dtype=np.float)
		
		for i in range(batch_size):
			x_in[i,0:len(data_input)] = data_input
		
		
		
		base_pred = self.base_model.predict(x = x_in)
		print('base_pred:\n', base_pred)
		
		
		y_p = base_pred
		print('base_pred0:\n',base_pred[0][0].shape)
		
		for j in range(200):
			mean = np.sum(y_p[0][j]) / y_p[0][j].shape[0]
			print('max y_p:',np.max(y_p[0][j]),'min y_p:',np.min(y_p[0][j]),'mean y_p:',mean,'mid y_p:',y_p[0][j][100])
			print('argmin:',np.argmin(y_p[0][j]),'argmax:',np.argmax(y_p[0][j]))
			count=0
			for i in range(y_p[0][j].shape[0]):
				if(y_p[0][j][i] < mean):
					count += 1
			print('count:',count)
		#for j in range(0,200):
		#	mean = sum(y_p[0][0][j]) / len(y_p[0][0][j])
		#	print('max y_p:',max(y_p[0][0][j]),'min y_p:',min(y_p[0][0][j]),'mean y_p:',mean,'mid y_p:',y_p[0][0][j][100])
		#	print('argmin:',np.argmin(y_p[0][0][j]),'argmax:',np.argmax(y_p[0][0][j]))
		#	count=0
		#	for i in y_p[0][0][j]:
		#		if(i < mean):
		#			count += 1
		#	print('count:',count)
		#decoded_sequences = self.decoder([base_pred, in_len])
		
		#print('decoded_sequences:\n', decoded_sequences)
		#input_length = tf.squeeze(input_length)
		
		#decode_pred = self.model_decode(x=[x_in, in_len])
		#print(decode_pred)
		base_pred =base_pred[:, 2:, :]
		r = K.ctc_decode(base_pred, in_len, greedy = True, beam_width=100, top_paths=1)
		print('r', r)
		#r = K.cast(r[0][0], dtype='float32')
		#print('r1', r)
		#print('解码完成')
		
		r1 = K.get_value(r[0][0])
		print('r1', r1)
		
		print('r0', r[1])
		r2 = K.get_value(r[1])
		print(r2)
		print('解码完成')
		list_symbol_dic = data.list_symbol # 获取拼音列表
		
		print('解码完成')
		return r1
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		#data = self.data
		#data = DataSpeech('E:\\语音数据集')
		#data.LoadDataList('dev')
		# 获取输入特征
		#data_input = data.GetMfccFeature(wavsignal, fs)
		#data_input = data.GetFrequencyFeature(wavsignal, fs)
		
		#arr_zero = np.zeros((1, 200), dtype=np.int16) #一个全是0的行向量
		
		#import matplotlib.pyplot as plt
		#plt.subplot(111)
		#plt.imshow(data_input, cmap=plt.get_cmap('gray'))
		#plt.show()
		
		#while(len(data_input)<1600): #长度不够时补全到1600
		#	data_input = np.row_stack((data_input,arr_zero))
		#print(len(data_input))
		
		#list_symbol = data.list_symbol # 获取拼音列表
		
		#labels = [ list_symbol[0] ]
		#while(len(labels) < 64):
		#	labels.append('')
			
		#labels_num = []
		#for i in labels:
		#	labels_num.append(data.SymbolToNum(i))
		
		
		
		#data_input = np.array(data_input, dtype=np.int16)
		#data_input = data_input.reshape(data_input.shape[0],data_input.shape[1])
		
		#labels_num = np.array(labels_num, dtype=np.int16)
		#labels_num = labels_num.reshape(labels_num.shape[0])
		
		#input_length = np.array([data_input.shape[0] // 4 - 3], dtype=np.int16)
		#input_length = np.array(input_length)
		#input_length = input_length.reshape(input_length.shape[0])
		
		#label_length = np.array([labels_num.shape[0]], dtype=np.int16)
		#label_length = np.array(label_length)
		#label_length = label_length.reshape(label_length.shape[0])
		
		#x = [data_input, labels_num, input_length, label_length]
		#x = next(data.data_genetator(1, self.AUDIO_LENGTH))
		#x = kr.utils.np_utils.to_categorical(x)
		
		#print(x)
		#x=np.array(x)
		
		#pred = self._model.predict(x=x)
		#pred = self._model.predict_on_batch([data_input, labels_num, input_length, label_length])
		#return [labels,pred]
		
		pass
		
	def RecognizeSpeech_FromFile(self, filename):
		'''
		最终做语音识别用的函数，识别指定文件名的语音
		'''
		
		wavsignal,fs = read_wav_data(filename)
		return self.RecognizeSpeech(wavsignal, fs)
		
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
	
	ms.LoadModel(modelpath + '5test\\speech_model_e_0_step_1400.model')
	#ms.TrainModel(datapath, epoch = 2, batch_size = 8, save_step = 10)
	#ms.TestModel(datapath, str_dataset='dev', data_count = 32)
	r = ms.RecognizeSpeech_FromFile('E:\\语音数据集\\wav\\test\\D4\\D4_750.wav')
	print('*[提示] 语音识别结果：\n',r)
