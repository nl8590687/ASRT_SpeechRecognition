#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import wave
import numpy as np
import matplotlib.pyplot as plt  

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
	time = np.arange(0, num_frame) * (1.0/framerate)  # 计算声音的播放时间，单位为秒
	return wave_data, time  
	
def wav_show(wave_data, time): # 显示出来声音波形
	#wave_data, time = read_wave_data("C:\\Users\\nl\\Desktop\\A2_0.wav")     
	#draw the wave  
	#plt.subplot(211)  
	plt.plot(time, wave_data[0])  
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
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split(' ')
			dic_filelist[txt_l[0]]=txt_l[1]
	return dic_filelist
	
def get_wav_symbol(filename):
	'''
	读取指定数据集中，所有wav文件对应的语音符号
	返回一个存储符号集的字典类型值
	'''
	print('test')
#if(__name__=='__main__'):
	#dic=get_wav_list('E:\\语音数据集\\doc\\doc\\list\\train.wav.lst')
	#for i in dic:
		#print(i,dic[i])
	#wave_data, time = read_wav_data("C:\\Users\\nl\\Desktop\\A2_0.wav")  
	#wav_show(wave_data,time)
	
