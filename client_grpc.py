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
ASRT语音识别asrserver grpc协议测试专用客户端
"""

import grpc
import time

from assets.asrt_pb2_grpc import AsrtGrpcServiceStub
from assets.asrt_pb2 import SpeechRequest, LanguageRequest, WavData

from utils.ops import read_wav_bytes


def run_speech():
    """
    请求ASRT服务Speech方法
    :return:
    """
    conn = grpc.insecure_channel('127.0.0.1:20002')
    client = AsrtGrpcServiceStub(channel=conn)

    wav_bytes, sample_rate, channels, sample_width = read_wav_bytes('assets/A11_0.wav')
    print('sample_width:', sample_width)
    wav_data = WavData(samples=wav_bytes, sample_rate=sample_rate,
                       channels=channels, byte_width=sample_width)

    request = SpeechRequest(wav_data=wav_data)
    time_stamp0 = time.time()
    response = client.Speech(request)
    time_stamp1 = time.time()
    print('time:', time_stamp1 - time_stamp0, 's')
    print("received:", response.result_data)


def run_lan():
    """
    请求ASRT服务Language方法
    :return:
    """
    conn = grpc.insecure_channel('127.0.0.1:20002')
    client = AsrtGrpcServiceStub(channel=conn)
    pinyin_data = ['ni3', 'hao3', 'ya5']
    request = LanguageRequest(pinyins=pinyin_data)
    time_stamp0 = time.time()
    response = client.Language(request)
    time_stamp1 = time.time()
    print('time:', time_stamp1 - time_stamp0, 's')
    print("received:", response.text_result)


def run_all():
    """
    请求ASRT服务All方法
    :return:
    """
    conn = grpc.insecure_channel('127.0.0.1:20002')
    client = AsrtGrpcServiceStub(channel=conn)

    wav_bytes, sample_rate, channels, sample_width = read_wav_bytes('assets/A11_0.wav')
    print('sample_width:', sample_width)
    wav_data = WavData(samples=wav_bytes, sample_rate=sample_rate,
                       channels=channels, byte_width=sample_width)

    request = SpeechRequest(wav_data=wav_data)
    time_stamp0 = time.time()
    response = client.All(request)
    time_stamp1 = time.time()
    print("received:", response.text_result)
    print('time:', time_stamp1 - time_stamp0, 's')


def run_stream():
    """
    请求ASRT服务Stream方法
    :return:
    """
    conn = grpc.insecure_channel('127.0.0.1:20002')
    client = AsrtGrpcServiceStub(channel=conn)

    wav_bytes, sample_rate, channels, sample_width = read_wav_bytes('assets/A11_0.wav')
    print('sample_width:', sample_width)
    wav_data = WavData(samples=wav_bytes, sample_rate=sample_rate,
                       channels=channels, byte_width=sample_width)

    # 先制造一些客户端能发送的数据
    def make_some_data():
        for _ in range(1):
            time.sleep(1)
            yield SpeechRequest(wav_data=wav_data)

    try:
        status_response = client.Stream(make_some_data())
        for ret in status_response:
            print("received:", ret.text_result, " , status:", ret.status_code)
            time.sleep(0.1)
    except Exception as any_exception:
        print(f'err in send_status:{any_exception}')
        return


if __name__ == '__main__':
    # run_all()
    run_stream()
