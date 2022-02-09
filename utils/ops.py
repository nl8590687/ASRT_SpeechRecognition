#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2016-2099 Ailemon.net
#
# This file is part of ASRT Speech Recognition Tool.
#
# ASRT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ASRT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASRT.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

"""
@author: nl8590687
一些常用操作函数的定义
"""

import wave
import difflib
import matplotlib.pyplot as plt
import numpy as np

def read_wav_data(filename: str) -> tuple:
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
    return wave_data, framerate, num_channel, num_sample_width

def read_wav_bytes(filename: str) -> tuple:
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
    return str_data, framerate, num_channel, num_sample_width

def get_edit_distance(str1, str2) -> int:
    '''
    计算两个串的编辑距离，支持str和list类型
    '''
    leven_cost = 0
    sequence_match = difflib.SequenceMatcher(None, str1, str2)
    for tag, index_1, index_2, index_j1, index_j2 in sequence_match.get_opcodes():
        if tag == 'replace':
            leven_cost += max(index_2-index_1, index_j2-index_j1)
        elif tag == 'insert':
            leven_cost += (index_j2-index_j1)
        elif tag == 'delete':
            leven_cost += (index_2-index_1)
    return leven_cost

def ctc_decode_delete_tail_blank(ctc_decode_list):
    '''
    处理CTC解码后序列末尾余留的空白元素，删除掉
    '''
    p = 0
    while p < len(ctc_decode_list) and ctc_decode_list[p] != -1:
        p += 1
    return ctc_decode_list[0:p]

def visual_1D(points_list, frequency=1):
    '''
    可视化1D数据
    '''
    # 首先创建绘图网格，1个子图
    fig, ax = plt.subplots(1)
    x = np.linspace(0, len(points_list)-1, len(points_list)) / frequency

    # 在对应对象上调用 plot() 方法
    ax.plot(x, points_list)
    fig.show()

def visual_2D(img):
    '''
    可视化2D数据
    '''
    plt.subplot(111)
    plt.imshow(img)
    plt.colorbar(cax=None, ax=None, shrink=0.5)
    plt.show() 

def decode_wav_bytes(samples_data: bytes, channels: int = 1, byte_width: int = 2) -> list:
    '''
    解码wav格式样本点字节流，得到numpy数组
    '''
    numpy_type = np.short
    if byte_width == 4:
        numpy_type = np.int
    elif byte_width != 2:
        raise Exception('error: unsurpport byte width `' + str(byte_width) + '`')
    wave_data = np.fromstring(samples_data, dtype = numpy_type) # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, channels # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T # 将矩阵转置
    return wave_data

def get_symbol_dict(dict_filename):
    '''
    读取拼音汉字的字典文件
    返回读取后的字典
    '''
    txt_obj = open(dict_filename, 'r', encoding='UTF-8') # 打开文件并读入
    txt_text = txt_obj.read()
    txt_obj.close()
    txt_lines = txt_text.split('\n') # 文本分割

    dic_symbol = {} # 初始化符号字典
    for i in txt_lines:
        list_symbol=[] # 初始化符号列表
        if i!='':
            txt_l=i.split('\t')
            pinyin = txt_l[0]
            for word in txt_l[1]:
                list_symbol.append(word)
        dic_symbol[pinyin] = list_symbol

    return dic_symbol

def get_language_model(model_language_filename):
    '''
    读取语言模型的文件
    返回读取后的模型
    '''
    txt_obj = open(model_language_filename, 'r', encoding='UTF-8') # 打开文件并读入
    txt_text = txt_obj.read()
    txt_obj.close()
    txt_lines = txt_text.split('\n') # 文本分割

    dic_model = {} # 初始化符号字典
    for i in txt_lines:
        if i!='':
            txt_l=i.split('\t')
            if len(txt_l) == 1:
                continue
            dic_model[txt_l[0]] = txt_l[1]

    return dic_model
