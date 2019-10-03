#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform as plat
import os

import numpy as np
from general_function.file_wav import *
from general_function.file_dict import *

import random
#import scipy.io.wavfile as wav
from scipy.fftpack import fft

class DataSpeech():
	
	
	def __init__(self, path, type, LoadToMem = False, MemWavCount = 10000):
		'''
		初始化
		参数：
			path：数据存放位置根目录
		'''
		
		self.datapath = path; # 数据存放位置根目录
		self.type = type # 数据类型，分为三种：训练集(train)、验证集(dev)、测试集(test)	
		
		self.dic_wavlist_thchs30 = {}
		self.dic_symbollist_thchs30 = {}
		self.dic_wavlist_stcmds = {}
		self.dic_symbollist_stcmds = {}
		
		self.SymbolNum = 0 # 记录拼音符号数量
		self.list_symbol = self.GetSymbolList() # 全部汉语拼音符号列表
		self.list_wavnum=[] # wav文件标记列表
		self.list_symbolnum=[] # symbol标记列表
		
		self.DataNum = 0 # 记录数据量
		self.LoadDataList()
		
		self.wavs_data = []
		self.LoadToMem = LoadToMem
		self.MemWavCount = MemWavCount
		pass
	
	def LoadDataList(self):
		'''
		加载用于计算的数据列表
		参数：
			type：选取的数据集类型
				train 训练集
				dev 开发集
				test 测试集
		'''
		# 设定选取哪一项作为要使用的数据集
		if(self.type=='train'):
			filename_wavlist_thchs30 = os.path.join('thchs30', 'train.wav.lst')
			filename_wavlist_stcmds = os.path.join('st-cmds', 'train.wav.txt')
			filename_symbollist_thchs30 = os.path.join('thchs30', 'train.syllable.txt')
			filename_symbollist_stcmds = os.path.join('st-cmds', 'train.syllable.txt')
		elif(self.type=='dev'):
			filename_wavlist_thchs30 = os.path.join('thchs30', 'cv.wav.lst')
			filename_wavlist_stcmds = os.path.join('st-cmds', 'dev.wav.txt')
			filename_symbollist_thchs30 = os.path.join('thchs30', 'cv.syllable.txt')
			filename_symbollist_stcmds = os.path.join('st-cmds', 'dev.syllable.txt')
		elif(self.type=='test'):
			filename_wavlist_thchs30 = os.path.join('thchs30', 'test.wav.lst')
			filename_wavlist_stcmds = os.path.join('st-cmds', 'test.wav.txt')
			filename_symbollist_thchs30 = os.path.join('thchs30', 'test.syllable.txt')
			filename_symbollist_stcmds = os.path.join('st-cmds', 'test.syllable.txt')
		else:
			filename_wavlist = '' # 默认留空
			filename_symbollist = ''
		# 读取数据列表，wav文件列表和其对应的符号列表
		self.dic_wavlist_thchs30,self.list_wavnum_thchs30 = get_wav_list(os.path.join(self.datapath, filename_wavlist_thchs30))
		self.dic_wavlist_stcmds,self.list_wavnum_stcmds = get_wav_list(os.path.join(self.datapath, filename_wavlist_stcmds))
		
		self.dic_symbollist_thchs30,self.list_symbolnum_thchs30 = get_wav_symbol(os.path.join(self.datapath, filename_symbollist_thchs30))
		self.dic_symbollist_stcmds,self.list_symbolnum_stcmds = get_wav_symbol(os.path.join(self.datapath, filename_symbollist_stcmds))
		self.DataNum = self.GetDataNum()
	
	def GetDataNum(self):
		'''
		获取数据的数量
		当wav数量和symbol数量一致的时候返回正确的值，否则返回-1，代表出错。
		'''
		num_wavlist_thchs30 = len(self.dic_wavlist_thchs30)
		num_symbollist_thchs30 = len(self.dic_symbollist_thchs30)
		num_wavlist_stcmds = len(self.dic_wavlist_stcmds)
		num_symbollist_stcmds = len(self.dic_symbollist_stcmds)
		if(num_wavlist_thchs30 == num_symbollist_thchs30 and num_wavlist_stcmds == num_symbollist_stcmds):
			DataNum = num_wavlist_thchs30 + num_wavlist_stcmds
		else:
			DataNum = -1
		
		return DataNum
		
		
	def GetData(self,n_start,n_amount=1):
		'''
		读取数据，返回神经网络输入值和输出值矩阵(可直接用于神经网络训练的那种)
		参数：
			n_start：从编号为n_start数据开始选取数据
			n_amount：选取的数据数量，默认为1，即一次一个wav文件
		返回：
			三个包含wav特征矩阵的神经网络输入值，和一个标定的类别矩阵神经网络输出值
		'''
		bili = 2
		if(self.type=='train'):
			bili = 11
			
		# 读取一个文件
		if(n_start % bili == 0):
			filename = self.dic_wavlist_thchs30[self.list_wavnum_thchs30[n_start // bili]]
			list_symbol=self.dic_symbollist_thchs30[self.list_symbolnum_thchs30[n_start // bili]]
		else:
			n = n_start // bili * (bili - 1)
			yushu = n_start % bili
			length=len(self.list_wavnum_stcmds)
			filename = self.dic_wavlist_stcmds[self.list_wavnum_stcmds[(n + yushu - 1)%length]]
			list_symbol=self.dic_symbollist_stcmds[self.list_symbolnum_stcmds[(n + yushu - 1)%length]]
		
		wavsignal,fs=read_wav_data(os.path.join(self.datapath, filename))
		
		# 获取输出特征
		
		feat_out=[]
		#print("数据编号",n_start,filename)
		for i in list_symbol:
			if(''!=i):
				n=self.SymbolToNum(i)
				#v=self.NumToVector(n)
				#feat_out.append(v)
				feat_out.append(n)
		#print('feat_out:',feat_out)
		
		# 获取输入特征
		data_input = GetFrequencyFeature3(wavsignal,fs)
		#data_input = np.array(data_input)
		data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
		#arr_zero = np.zeros((1, 39), dtype=np.int16) #一个全是0的行向量
		
		#while(len(data_input)<1600): #长度不够时补全到1600
		#	data_input = np.row_stack((data_input,arr_zero))
		
		#data_input = data_input.T
		data_label = np.array(feat_out)
		return data_input, data_label
	
	def data_genetator(self, batch_size=32, audio_length = 1600):
		'''
		数据生成器函数，用于Keras的generator_fit训练
		batch_size: 一次产生的数据量
		需要再修改。。。
		'''
		
		#labels = []
		#for i in range(0,batch_size):
		#	#input_length.append([1500])
		#	labels.append([0.0])
		
		
		
		#labels = np.array(labels, dtype = np.float)
		labels = np.zeros((batch_size,1), dtype = np.float)
		#print(input_length,len(input_length))
		
		while True:
			X = np.zeros((batch_size, audio_length, 200, 1), dtype = np.float)
			#y = np.zeros((batch_size, 64, self.SymbolNum), dtype=np.int16)
			y = np.zeros((batch_size, 64), dtype=np.int16)
			
			#generator = ImageCaptcha(width=width, height=height)
			input_length = []
			label_length = []
			
			
			
			for i in range(batch_size):
				ran_num = random.randint(0,self.DataNum - 1) # 获取一个随机数
				data_input, data_labels = self.GetData(ran_num)  # 通过随机数取一个数据
				#data_input, data_labels = self.GetData((ran_num + i) % self.DataNum)  # 从随机数开始连续向后取一定数量数据
				
				# 关于下面这一行取整除以8 并加8的余数，在实际中如果遇到报错，可尝试只在有余数时+1，没有余数时+0，或者干脆都不加，只留整除
				input_length.append(data_input.shape[0] // 8 + data_input.shape[0] % 8)
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
			
			label_length = np.matrix(label_length)
			input_length = np.array([input_length]).T
			#input_length = np.array(input_length)
			#print('input_length:\n',input_length)
			#X=X.reshape(batch_size, audio_length, 200, 1)
			#print(X)
			yield [X, y, input_length, label_length ], labels
		pass
		
	def GetSymbolList(self):
		'''
		加载拼音符号列表，用于标记符号
		返回一个列表list类型变量
		'''
		txt_obj=open('dict.txt','r',encoding='UTF-8') # 打开文件并读入
		txt_text=txt_obj.read()
		txt_lines=txt_text.split('\n') # 文本分割
		list_symbol=[] # 初始化符号列表
		for i in txt_lines:
			if(i!=''):
				txt_l=i.split('\t')
				list_symbol.append(txt_l[0])
		txt_obj.close()
		list_symbol.append('_')
		self.SymbolNum = len(list_symbol)
		return list_symbol

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
	#print(l.GetData(0))
	#aa=l.data_genetator()
	#for i in aa:
		#a,b=i
	#print(a,b)
	pass
	