#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: nl8590687
asrserver测试专用客户端

'''

import requests
from general_function.file_wav import *

url = 'http://127.0.0.1:20000/'

token = 'qwertasd'

#wavsignal,fs=read_wav_data('E:\\语音数据集\\ST-CMDS-20170001_1-OS\\20170001P00241I0052.wav')
wavsignal,fs=read_wav_data('C:\\Users\\nl\\Desktop\\20180506_114631.wav')
#wavsignal,fs=read_wav_data('E:\\国创项目工程\\代码\\语音识别笔记本UWP\\语音识别笔记本UWP\\bin\\x86\\Debug\\AppX\\12345.wav')

#print(wavsignal,fs)

datas={'token':token, 'fs':fs, 'wavs':wavsignal}

r = requests.post(url, datas)

r.encoding='utf-8'

print(r.text)