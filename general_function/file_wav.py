#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import wave
import numpy as np
import matplotlib.pyplot as plt  
import math

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
	
def wav_scale(energy):
	'''
	语音信号能量归一化
	'''
	means = energy.mean() # 均值
	var=energy.var() # 方差
	e=(energy-means)/math.sqrt(var) # 归一化能量
	return e
	
def wav_show(wave_data, fs): # 显示出来声音波形
	time = np.arange(0, len(wave_data)) * (1.0/fs)  # 计算声音的播放时间，单位为秒
	# 画声音波形
	#plt.subplot(211)  
	plt.plot(time, wave_data)  
	#plt.subplot(212)  
	#plt.plot(time, wave_data[1], c = "g")  
	plt.show()  

	
def get_wav_list(filename):
	'''
	读取一个wav文件列表，返回一个存储该列表的字典类型值
	ps:在数据中专门有几个文件用于存放用于训练、验证和测试的wav文件列表
	'''
	txt_obj=open(filename,'r') # 打开文件并读入
	txt_text=txt_obj.read()
	txt_lines=txt_text.split('\n') # 文本分割
	dic_filelist={} # 初始化字典
	list_wavmark=[] # 初始化wav列表
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split(' ')
			dic_filelist[txt_l[0]] = txt_l[1]
			list_wavmark.append(txt_l[0])
	txt_obj.close()
	return dic_filelist,list_wavmark
	
def get_wav_symbol(filename):
	'''
	读取指定数据集中，所有wav文件对应的语音符号
	返回一个存储符号集的字典类型值
	'''
	txt_obj=open(filename,'r') # 打开文件并读入
	txt_text=txt_obj.read()
	txt_lines=txt_text.split('\n') # 文本分割
	dic_symbol_list={} # 初始化字典
	list_symbolmark=[] # 初始化symbol列表
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split(' ')
			dic_symbol_list[txt_l[0]]=txt_l[1:]
			list_symbolmark.append(txt_l[0])
	txt_obj.close()
	return dic_symbol_list,list_symbolmark
	
if(__name__=='__main__'):
	#dic=get_wav_symbol('E:\\语音数据集\\doc\\doc\\trans\\train.syllable.txt')
	#print(dic)
	#dic=get_wav_list('E:\\语音数据集\\doc\\doc\\list\\train.wav.lst')
	#for i in dic:
		#print(i,dic[i])
	wave_data, fs = read_wav_data("A2_0.wav")  
	#wave_data[0]=wav_scale(wave_data[0])
	#print(fs)
	wav_show(wave_data[0],fs)
	
