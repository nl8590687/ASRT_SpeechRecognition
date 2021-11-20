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

'''
@author: nl8590687
ASRT语音识别asrserver测试专用客户端
'''
import time
import requests
from utils.ops import read_wav_data

URL = 'http://127.0.0.1:20000/'
TOKEN = 'qwertasd'
wavsignal, fs, _, _ = read_wav_data('X:\\语音数据集\\data_thchs30\\train\\A11_0.wav')
datas={'token':TOKEN, 'fs':fs, 'wavs':wavsignal}

t0=time.time()
r = requests.post(URL, datas)
t1=time.time()
r.encoding='utf-8'

print(r.text)
print('time:', t1-t0, 's')
