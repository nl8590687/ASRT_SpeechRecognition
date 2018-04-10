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
		

print('main')
pass