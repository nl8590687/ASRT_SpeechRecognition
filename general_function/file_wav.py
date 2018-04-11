#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import wave
import numpy as np
import matplotlib.pyplot as plt  
import math

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

from scipy.fftpack import fft

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

def GetMfccFeature(wavsignal, fs):
	# 获取输入特征
	feat_mfcc=mfcc(wavsignal[0],fs)
	feat_mfcc_d=delta(feat_mfcc,2)
	feat_mfcc_dd=delta(feat_mfcc_d,2)
	# 返回值分别是mfcc特征向量的矩阵及其一阶差分和二阶差分矩阵
	wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
	return wav_feature

def GetFrequencyFeature(wavsignal, fs):
	# wav波形 加时间窗以及时移10ms
	time_window = 25 # 单位ms
	data_input = []
	
	#print(int(len(wavsignal[0])/fs*1000 - time_window) // 10)
	for i in range(0,int(len(wavsignal[0])/fs*1000 - time_window) // 10 ):
		p_start = i * 160
		p_end = p_start + 400
		data_line = []
		
		for j in range(p_start, p_end):
			data_line.append(wavsignal[0][j])
			#print('wavsignal[0][j]:\n',wavsignal[0][j])
		#data_line = abs(fft(data_line)) / len(wavsignal[0])
		data_line = fft(data_line) / len(wavsignal[0])
		data_input.append(data_line[0:len(data_line)//2]) # 除以2是取一半数据，因为是对称的
		#print('data_line:\n',data_line)
	return data_input

def wav_scale(energy):
	'''
	语音信号能量归一化
	'''
	means = energy.mean() # 均值
	var=energy.var() # 方差
	e=(energy-means)/math.sqrt(var) # 归一化能量
	return e

def wav_scale2(energy):
	'''
	语音信号能量归一化
	'''
	maxnum = max(energy)
	e = energy / maxnum
	return e

def wav_scale3(energy):
	'''
	语音信号能量归一化
	'''
	for i in range(len(energy)):
		#if i == 1:
		#	#print('wavsignal[0]:\n {:.4f}'.format(energy[1]),energy[1] is int)
		energy[i] = float(energy[i]) / 100.0
		#if i == 1:
		#	#print('wavsignal[0]:\n {:.4f}'.format(energy[1]),energy[1] is int)
	return energy
	
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
	
