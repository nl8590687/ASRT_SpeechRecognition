#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from general_function.file_wav import *

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

import random
#import scipy.io.wavfile as wav

class DataSpeech():
	
	
	def __init__(self,path):
		'''
		初始化
		参数：
			path：数据存放位置根目录
		'''
		self.datapath = path; # 数据存放位置根目录
		if('\\'!=self.datapath[-1]): # 在目录路径末尾增加斜杠
			self.datapath=self.datapath+'\\'
		self.dic_wavlist = {}
		self.dic_symbollist = {}
		self.SymbolNum = 0 # 记录拼音符号数量
		self.list_symbol = self.GetSymbolList() # 全部汉语拼音符号列表
		self.list_wavnum=[] # wav文件标记列表
		self.list_symbolnum=[] # symbol标记列表
		
		self.DataNum = 0 # 记录数据量
		
		pass
	
	def LoadDataList(self,type):
		'''
		加载用于计算的数据列表
		参数：
			type：选取的数据集类型
				train 训练集
				dev 开发集
				test 测试集
		'''
		# 设定选取哪一项作为要使用的数据集
		if(type=='train'):
			filename_wavlist='doc\\doc\\list\\train.wav.lst'
			filename_symbollist='doc\\doc\\trans\\train.syllable.txt'
		elif(type=='dev'):
			filename_wavlist='doc\\doc\\list\\cv.wav.lst'
			filename_symbollist='doc\\doc\\trans\\cv.syllable.txt'
		elif(type=='test'):
			filename_wavlist='doc\\doc\\list\\test.wav.lst'
			filename_symbollist='doc\\doc\\trans\\test.syllable.txt'
		else:
			filename_wavlist='' # 默认留空
			filename_symbollist=''
		# 读取数据列表，wav文件列表和其对应的符号列表
		self.dic_wavlist,self.list_wavnum = get_wav_list(self.datapath+filename_wavlist)
		self.dic_symbollist,self.list_symbolnum = get_wav_symbol(self.datapath+filename_symbollist)
		self.DataNum = self.GetDataNum()
	
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
		
	def GetData(self,n_start,n_amount=1):
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
		
		filename=filename.replace('/','\\') # windows系统下需要添加这一行
		
		wavsignal,fs=read_wav_data(self.datapath+filename)
		# 获取输入特征
		feat_mfcc=mfcc(wavsignal[0],fs)
		feat_mfcc_d=delta(feat_mfcc,2)
		feat_mfcc_dd=delta(feat_mfcc_d,2)
		# 获取输出特征
		list_symbol=self.dic_symbollist[self.list_symbolnum[n_start]]
		feat_out=[]
		#print("数据编号",n_start,filename)
		for i in list_symbol:
			if(''!=i):
				n=self.SymbolToNum(i)
				v=self.NumToVector(n)
				feat_out.append(v)
		#print('feat_out:',feat_out)
		# 返回值分别是mfcc特征向量的矩阵及其一阶差分和二阶差分矩阵，以及对应的拼音符号矩阵
		data_input = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
		data_label = np.array(feat_out)
		return data_input, data_label
	
	def data_genetator(self, batch_size=32):
		'''
		数据生成器函数，用于Keras的generator_fit训练
		batch_size: 一次产生的数据量
		需要再修改。。。
		'''
		X = np.zeros((batch_size, 1500,39), dtype=np.int16)
		y = np.zeros((batch_size, 60, self.SymbolNum), dtype=np.int16)
		while True:
			#generator = ImageCaptcha(width=width, height=height)
			ran_num = random.randint(0,self.DataNum - 1) # 获取一个随机数
			for i in range(batch_size):
				data_input, data_labels = self.GetData((ran_num + i) % self.DataNum)  # 从随机数开始连续向后取一定数量数据
				#print(data_input, data_labels)
				#print('data_input长度:',len(data_input))
				X[i,0:len(data_input)] = data_input
				#print('data_labels长度:',len(data_labels))
				#print(data_labels)
				y[i,0:len(data_labels)] = data_labels
			yield X, y
		pass
		
	def GetSymbolList(self):
		'''
		加载拼音符号列表，用于标记符号
		返回一个列表list类型变量
		'''
		txt_obj=open(self.datapath+'dict.txt','r',encoding='UTF-8') # 打开文件并读入
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
		return self.list_symbol.index(symbol)
	
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
	#wave_data, fs = read_wav_data("general_function\\A2_0.wav")  
	#print(wave_data)
	#(fs,wave_data)=wav.read('E:\\国创项目工程\代码\\ASRT_SpeechRecognition\\general_function\\A2_0.wav')
	#wav_show(wave_data[0],fs)
	#mfcc_feat = mfcc(wave_data[0],fs) # 计算MFCC特征
	#print(mfcc_feat[0:3,:])
	#d_mfcc_feat_1 = delta(mfcc_feat, 2)
	#print(d_mfcc_feat_1[0,:])
	#d_mfcc_feat_2 = delta(d_mfcc_feat_1, 2)
	#print(d_mfcc_feat_2[0,:])
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
	