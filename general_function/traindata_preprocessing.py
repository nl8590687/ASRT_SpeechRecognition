#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
数据预处理用的程序
'''

import os
import wave
import numpy as np

def read_wav_data(filename):
	'''
	读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
	'''
	wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
	num_frame = wav.getnframes() # 获取帧数
	num_channel=wav.getnchannels() # 获取声道数
	framerate=wav.getframerate() # 获取帧速率
	num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
	str_data = wav.readframes(num_frame) # 读取全部的帧
	wav.close() # 关闭流
	wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
	wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
	wave_data = wave_data.T # 将矩阵转置
	#wave_data = wave_data 
	return wave_data, framerate  
	
def GetFrequencyFeature(self, wavsignal, fs):
	# wav波形 加时间窗以及时移10ms
	time_window = 25 # 单位ms
	data_input = []
	
	for i in range(0,int(len(wavsignal[0])/fs*1000 - time_window) // 10 ):
		p_start = i * 160
		p_end = p_start + 400
		data_line = []
		
		for j in range(p_start, p_end):
			data_line.append(wavsignal[0][j])
		
		data_line = fft(data_line) / len(wavsignal[0])
		data_input.append(data_line[0:len(data_line)//2]) # 除以2是取一半数据，因为是对称的
		
	return data_input
		
def get_wav_list(filename):
	'''
	读取一个wav文件列表，返回一个存储该列表的字典类型值
	ps:在数据中专门有几个文件用于存放用于训练、验证和测试的wav文件列表
	'''
	txt_obj=open(filename,'r') # 打开文件并读入
	txt_text=txt_obj.read()
	txt_lines=txt_text.split('\n') # 文本分割
	dic_filelist={} # 初始化字典
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split(' ')
			dic_filelist[txt_l[0]] = txt_l[1]
	txt_obj.close()
	return dic_filelist

def GetWavDataList(type):
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
		filename_wavlist = 'doc\\list\\train.wav.lst'
	elif(type=='dev'):
		filename_wavlist = 'doc\\list\\cv.wav.lst'
	elif(type=='test'):
		filename_wavlist = 'doc\\list\\test.wav.lst'
	else:
		filename_wavlist = '' # 默认留空
	# 读取数据列表，wav文件列表和其对应的符号列表
	self.dic_wavlist = get_wav_list(self.datapath + filename_wavlist)
	return filename_wavlist

print('main')

pass