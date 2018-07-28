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

wavsignal,fs=read_wav_data('E:\\语音数据集\\ST-CMDS-20170001_1-OS\\20170001P00241I0052.wav')

#print(wavsignal,fs)

datas={'token':token, 'fs':fs, 'wavs':wavsignal}

r = requests.post(url, datas)

r.encoding='utf-8'

print(r.text)