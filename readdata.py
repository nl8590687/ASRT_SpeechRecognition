#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat

import numpy as np
from general_function.file_wav import *
from general_function.file_dict import *

import random
#import scipy.io.wavfile as wav


import matplotlib.pyplot as plt 

class DataSpeech():
	
	
	def __init__(self, path, type, LoadToMem = False, MemWavCount = 10000):
		'''
		初始化
		参数：
			path：数据存放位置根目录
			LoadToMem: 是否将大量数据一次性读入内存
			MemWavCount: 一次性载入内存的数据数量
			
			type：选取的数据集类型
				train 训练集
				dev 开发集
				test 测试集
		'''
		
		system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
		
		self.datapath = path; # 数据存放位置根目录
		self.type = type # 数据类型，分为三种：训练集(train)、验证集(dev)、测试集(test)
		
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
		
		self.dic_wavlist = {}
		self.dic_symbollist = {}
		
		self.list_symbol = GetSymbolList(self.datapath) # 全部汉语拼音符号列表
		self.SymbolNum = len(self.list_symbol) # 记录拼音符号数量
		
		self.list_wavnum = [] # wav文件标记列表
		self.list_symbolnum = [] # symbol标记列表
		
		self.DataNum = 0 # 记录数据量
		self.LoadDataList()
		
		self.wavs_data = []
		self.LoadToMem = LoadToMem
		self.MemWavCount = MemWavCount
		if(LoadToMem == True):
			print('*[提示] 正在准备将全部数据加载到内存...Count: ', MemWavCount)
			self.LoadWavData()
			pass
		pass
	
	def LoadDataList(self):
		'''
		加载用于计算的数据列表
		
		'''
		# 设定选取哪一项作为要使用的数据集
		if(self.type=='train'):
			filename_wavlist = 'doc' + self.slash + 'list' + self.slash + 'train.wav.lst'
			filename_symbollist = 'doc' + self.slash + 'trans' + self.slash + 'train.syllable.txt'
		elif(self.type=='dev'):
			filename_wavlist = 'doc' + self.slash + 'list' + self.slash + 'cv.wav.lst'
			filename_symbollist = 'doc' + self.slash + 'trans' + self.slash + 'cv.syllable.txt'
		elif(self.type=='test'):
			filename_wavlist = 'doc' + self.slash + 'list' + self.slash + 'test.wav.lst'
			filename_symbollist = 'doc' + self.slash + 'trans' + self.slash + 'test.syllable.txt'
		else:
			filename_wavlist = '' # 默认留空
			filename_symbollist = ''
		# 读取数据列表，wav文件列表和其对应的符号列表
		self.dic_wavlist,self.list_wavnum = get_wav_list(self.datapath + filename_wavlist)
		self.dic_symbollist,self.list_symbolnum = get_wav_symbol(self.datapath + filename_symbollist)
		self.DataNum = self.GetDataNum()
	
	def LoadWavData(self):
		'''
		将所有数据读入内存
		'''
		for i in range(self.MemWavCount):
			# 读取一个文件
			filename = self.dic_wavlist[self.list_wavnum[i]]
			
			if('Windows' == plat.system()):
				filename=filename.replace('/','\\') # windows系统下需要执行这一行，对文件路径做特别处理
		
			wavsignal,fs = read_wav_data(self.datapath+filename)
			self.wavs_data.append([wavsignal,fs])
			
			print('*[提示] 全部数据已经加载到内存')
		pass
		
	def GetDataNum(self):
		'''
		获取数据的数量
		当wav数量和symbol数量一致的时候返回正确的值，否则返回-1，代表出错。
		'''
		if(len(self.dic_wavlist) == len(self.dic_symbollist)):
			DataNum = len(self.dic_wavlist)
		else:
			DataNum = -1
		
		return DataNum
		
	def GetDataFromMem(self, n_start, n_amount = 1):
		'''
		从内存中的self.wavs_data里读取数据
		'''
		assert len(self.wavs_data) > 0
		
		[wavsignal,fs] = self.wavs_data[n_start]
		
		data_input = GetFrequencyFeature(wavsignal, fs)
		
		# 获取输出特征
		list_symbol=self.dic_symbollist[self.list_symbolnum[n_start]]
		feat_out=[]
		#print("数据编号",n_start,filename)
		for i in list_symbol:
			if(''!=i):
				n=self.SymbolToNum(i)
				feat_out.append(n)

		data_input = np.array(data_input)
		data_label = np.array(feat_out)
		return data_input, data_label
		pass
		
	def GetData(self, n_start, n_amount = 1):
		'''
		读取数据，返回神经网络输入值和输出值矩阵(可直接用于神经网络训练的那种)
		参数：
			n_start：从编号为n_start数据开始选取数据
			n_amount：选取的数据数量，默认为1，即一次一个wav文件
		返回：
			三个包含wav特征矩阵的神经网络输入值，和一个标定的类别矩阵神经网络输出值
		'''
		# 读取一个文件
		filename = self.dic_wavlist[self.list_wavnum[n_start]]
		
		if('Windows' == plat.system()):
			filename=filename.replace('/','\\') # windows系统下需要执行这一行，对文件路径做特别处理
		
		wavsignal,fs = read_wav_data(self.datapath+filename)
		
		data_input = GetFrequencyFeature(wavsignal, fs)
		
		#data_input = self.GetMfccFeature(wavsignal, fs)
		
		
		# 获取输出特征
		list_symbol=self.dic_symbollist[self.list_symbolnum[n_start]]
		feat_out=[]
		#print("数据编号",n_start,filename)
		for i in list_symbol:
			if(''!=i):
				n=self.SymbolToNum(i)
				#v=self.NumToVector(n)
				#feat_out.append(v)
				feat_out.append(n)
		#print('feat_out:',feat_out)
		# 获得对应的拼音符号向量
		
		
		#arr_zero = np.zeros((1, 39), dtype=np.int16) #一个全是0的行向量
		
		#while(len(data_input)<1600): #长度不够时补全到1600
		#	data_input = np.row_stack((data_input,arr_zero))
		
		#data_input = data_input.T
		data_input = np.array(data_input)
		data_label = np.array(feat_out)
		return data_input, data_label
	
	def data_genetator(self, batch_size=32, audio_length = 1600):
		'''
		数据生成器函数，用于Keras的generator_fit训练
		batch_size: 一次产生的数据量
		'''
		X = np.zeros((batch_size, audio_length, 200), dtype=np.float)
		#y = np.zeros((batch_size, 64, self.SymbolNum), dtype=np.int16)
		y = np.zeros((batch_size, 64), dtype=np.int16)
		
		
		
		labels = []
		for i in range(0,batch_size):
			#input_length.append([1500])
			labels.append([0]) # 最终的ctc loss结果，0代表着没有ctc上的loss
		
		
		
		#labels = np.matrix(labels)
		labels = np.array(labels, dtype = np.float)
		#print(input_length,len(input_length))
		
		while True:
			#generator = ImageCaptcha(width=width, height=height)
			input_length = []
			label_length = []
			
			ran_num = random.randint(0,self.DataNum - 1) # 获取一个随机数
			for i in range(batch_size):
				if(self.LoadToMem == False):
					data_input, data_labels = self.GetData((ran_num + i) % self.DataNum)  # 从随机数开始连续向后取一定数量数据
				else:
					data_input, data_labels = self.GetDataFromMem((ran_num + i) % self.DataNum)  # 从随机数开始连续向后取一定数量数据
					
				#data_input, data_labels = self.GetData(1 % self.DataNum)  # 从随机数开始连续向后取一定数量数据
				
				#input_length.append(data_input.shape[1] // 4 - 2)
				#print(data_input.shape[0],len(data_input))
				input_length.append(data_input.shape[0] // 4)
				#print(data_input, data_labels)
				#print('data_input长度:',len(data_input))
				
				X[i,0:len(data_input)] = data_input
				#print('data_labels长度:',len(data_labels))
				#print(data_labels)
				y[i,0:len(data_labels)] = data_labels
				#print(i,y[i].shape)
				#y[i] = y[i].T
				#print(i,y[i].shape)
				label_length.append([len(data_labels)])
			
			label_length = np.array(label_length)
			input_length = np.array(input_length).T
			yield [X, y, input_length, label_length ], labels
		pass
		
	

	def GetSymbolNum(self):
		'''
		获取拼音符号数量
		'''
		return len(self.list_symbol)
		
	def SymbolToNum(self,symbol):
		'''
		符号转为数字
		'''
		if(symbol != ''):
			return self.list_symbol.index(symbol)
		return self.SymbolNum
	
	def NumToVector(self,num):
		'''
		数字转为对应的向量
		'''
		v_tmp=[]
		for i in range(0,len(self.list_symbol)):
			if(i==num):
				v_tmp.append(1)
			else:
				v_tmp.append(0)
		v=np.array(v_tmp)
		return v
	
if(__name__=='__main__'):
	#path='E:\\语音数据集'
	#l=DataSpeech(path)
	#l.LoadDataList('train')
	#print(l.GetDataNum())
	#data0=l.GetData(0)
	#print(data0)
	#data0=data0[0].reshape(data0[0].shape[0],data0[0].shape[1])
	#print(data0, data0 is list)
	#plt.subplot(111)
	#plt.imshow(data0.T, cmap=plt.get_cmap('Blues_r'))
	#plt.show()
	#aa=l.data_genetator()
	#for i in aa:
		#a,b=i
	#print(a,b)
	pass
	