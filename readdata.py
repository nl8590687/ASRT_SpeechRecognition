#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from general_function.file_wav import *

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

#import scipy.io.wavfile as wav

class DataSpeech():
	def __init__(self,path):
		'''
		初始化
		参数：
			path：数据存放位置根目录
		'''
		pass
		
	def GetData(self,n):
		'''
		读取数据，返回神经网络输入值和输出值矩阵
		参数：
			n：第几个数据
		'''
		pass
	
	def GetDataNum(self):
		'''
		获取数据的数量
		'''
		pass
	
	
if(__name__=='__main__'):
	wave_data, fs = read_wav_data("general_function\\A2_0.wav")  
	print(wave_data)
	#(fs,wave_data)=wav.read('E:\\国创项目工程\代码\\ASRT_SpeechRecognition\\general_function\\A2_0.wav')
	wav_show(wave_data[0],fs)
	#mfcc_feat = mfcc(wave_data[0],fs) # 计算MFCC特征
	#print(mfcc_feat[100:110,:])
	#d_mfcc_feat_1 = delta(mfcc_feat, 2)
	#print(d_mfcc_feat_1[0,:])
	#d_mfcc_feat_2 = delta(d_mfcc_feat_1, 2)
	#print(d_mfcc_feat_2[0,:])
	pass
	