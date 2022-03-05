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
ASRT语音识别内置声学特征提取模块，定义了几个常用的声学特征类
"""

import random
import numpy as np
from scipy.fftpack import fft
from .base import mfcc, delta, logfbank

# 可用isinstance(object,class)来判断某对象是否属于某个类

class SpeechFeatureMeta():
    '''
    ASRT语音识别中所有声学特征提取类的基类
    '''
    def __init__(self, framesamplerate = 16000):
        self.framesamplerate = framesamplerate

    def run(self, wavsignal, fs = 16000):
        '''
        run method
        '''
        raise NotImplementedError('[ASRT] `run()` method is not implemented.')

class MFCC(SpeechFeatureMeta):
    '''
    ASRT语音识别内置的mfcc声学特征提取类

    Compute MFCC features from an audio signal.

    :param framesamplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    '''
    def __init__(self, framesamplerate = 16000,
                    winlen=0.025,
                    winstep=0.01,
                    numcep=13,
                    nfilt=26,
                    preemph=0.97):
        self.framesamplerate = framesamplerate
        self.winlen = winlen
        self.winstep = winstep
        self.numcep = numcep
        self.nfilt = nfilt
        self.preemph = preemph
        super().__init__(framesamplerate)

    def run(self, wavsignal, fs = 16000):
        '''
        计算mfcc声学特征，包含静态特征、一阶差分和二阶差分

        :returns: A numpy array of size (NUMFRAMES by numcep * 3) containing features. Each row holds 1 feature vector.
        '''
        wavsignal = np.array(wavsignal, dtype=np.float)
        # 获取输入特征
        feat_mfcc=mfcc(wavsignal[0], samplerate=self.framesamplerate, winlen=self.winlen,
            winstep=self.winstep, numcep=self.numcep, nfilt=self.nfilt, preemph=self.preemph)
        feat_mfcc_d=delta(feat_mfcc, 2)
        feat_mfcc_dd=delta(feat_mfcc_d, 2)
        # 返回值分别是mfcc特征向量的矩阵及其一阶差分和二阶差分矩阵
        wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
        return wav_feature

class Logfbank(SpeechFeatureMeta):
    '''
    ASRT语音识别内置的logfbank声学特征提取类
    '''
    def __init__(self, framesamplerate = 16000, nfilt=26):
        self.nfilt = nfilt
        super().__init__(framesamplerate)

    def run(self, wavsignal, fs = 16000):
        wavsignal = np.array(wavsignal, dtype=np.float)
        # 获取输入特征
        wav_feature = logfbank(wavsignal, fs, nfilt=self.nfilt)
        return wav_feature

class Spectrogram(SpeechFeatureMeta):
    '''
    ASRT语音识别内置的语谱图声学特征提取类
    '''
    def __init__(self, framesamplerate = 16000, timewindow = 25, timeshift = 10):
        self.time_window = timewindow
        self.window_length = int(framesamplerate / 1000 * self.time_window) # 计算窗长度的公式，目前全部为400固定值
        self.timeshift = timeshift

        '''
        # 保留将来用于不同采样频率
        self.x=np.linspace(0, self.window_length - 1, self.window_length, dtype = np.int64)
        self.w = 0.54 - 0.46 * np.cos(2 * np.pi * (self.x) / (self.window_length - 1) ) # 汉明窗
        '''

        self.x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
        self.w = 0.54 - 0.46 * np.cos(2 * np.pi * (self.x) / (400 - 1) ) # 汉明窗
        super().__init__(framesamplerate)

    def run(self, wavsignal, fs = 16000):
        if fs != 16000:
            raise ValueError('[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')

        # wav波形 加时间窗以及时移10ms
        time_window = 25 # 单位ms
        window_length = int(fs / 1000 * time_window) # 计算窗长度的公式，目前全部为400固定值

        wav_arr = np.array(wavsignal)
        #wav_length = len(wavsignal[0])
        #wav_length = wav_arr.shape[1]

        range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
        data_input = np.zeros((range0_end, window_length // 2), dtype = np.float) # 用于存放最终的频率特征数据
        data_line = np.zeros((1, window_length), dtype = np.float)

        for i in range(0, range0_end):
            p_start = i * 160
            p_end = p_start + 400

            data_line = wav_arr[0, p_start:p_end]
            data_line = data_line * self.w # 加窗
            data_line = np.abs(fft(data_line))

            data_input[i]=data_line[0: window_length // 2] # 设置为400除以2的值（即200）是取一半数据，因为是对称的

        #print(data_input.shape)
        data_input = np.log(data_input + 1)
        return data_input

class SpecAugment(SpeechFeatureMeta):
    '''
    复现谷歌SpecAugment数据增强特征算法，基于Spectrogram语谱图基础特征
    '''
    def __init__(self, framesamplerate = 16000, timewindow = 25, timeshift = 10):
        self.time_window = timewindow
        self.window_length = int(framesamplerate / 1000 * self.time_window) # 计算窗长度的公式，目前全部为400固定值
        self.timeshift = timeshift

        '''
        # 保留将来用于不同采样频率
        self.x=np.linspace(0, self.window_length - 1, self.window_length, dtype = np.int64)
        self.w = 0.54 - 0.46 * np.cos(2 * np.pi * (self.x) / (self.window_length - 1) ) # 汉明窗
        '''

        self.x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
        self.w = 0.54 - 0.46 * np.cos(2 * np.pi * (self.x) / (400 - 1) ) # 汉明窗
        super().__init__(framesamplerate)

    def run(self, wavsignal, fs = 16000):
        if fs != 16000:
            raise ValueError('[Error] ASRT currently only supports wav audio files with a sampling rate of 16000 Hz, but this audio is ' + str(fs) + ' Hz. ')

        # wav波形 加时间窗以及时移10ms
        time_window = 25 # 单位ms
        window_length = int(fs / 1000 * time_window) # 计算窗长度的公式，目前全部为400固定值

        wav_arr = np.array(wavsignal)
        #wav_length = len(wavsignal[0])
        #wav_length = wav_arr.shape[1]

        range0_end = int(len(wavsignal[0])/fs*1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
        data_input = np.zeros((range0_end, window_length // 2), dtype = np.float) # 用于存放最终的频率特征数据
        data_line = np.zeros((1, window_length), dtype = np.float)

        for i in range(0, range0_end):
            p_start = i * 160
            p_end = p_start + 400

            data_line = wav_arr[0, p_start:p_end]
            data_line = data_line * self.w # 加窗
            data_line = np.abs(fft(data_line))

            data_input[i]=data_line[0: window_length // 2] # 设置为400除以2的值（即200）是取一半数据，因为是对称的

        #print(data_input.shape)
        data_input = np.log(data_input + 1)

        # 开始对得到的特征应用SpecAugment
        mode = random.randint(1,100)
        h_start = random.randint(1,data_input.shape[0])
        h_width = random.randint(1,100)

        v_start = random.randint(1,data_input.shape[1])
        v_width = random.randint(1,100)

        if mode <= 60: # 正常特征 60%
            pass
        elif 60 < mode <=75: # 横向遮盖 15%
            data_input[h_start:h_start+h_width,:] = 0
        elif 75 < mode <= 90: # 纵向遮盖 15%
            data_input[:,v_start:v_start+v_width] = 0
        else: # 两种遮盖叠加 10%
            data_input[h_start:h_start+h_width,:v_start:v_start+v_width] = 0

        return data_input
