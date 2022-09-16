# !/usr/bin/env python3
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
ASRT语音识别基于gRPC协议的API服务器程序
"""

import argparse
import time
from concurrent import futures
import grpc

from assets.asrt_pb2_grpc import AsrtGrpcServiceServicer, add_AsrtGrpcServiceServicer_to_server
from assets.asrt_pb2 import SpeechResponse, TextResponse
from speech_model import ModelSpeech
from model_zoo.speech_model.keras_backend import SpeechModel251BN
from speech_features import Spectrogram
from language_model3 import ModelLanguage
from utils.ops import decode_wav_bytes

API_STATUS_CODE_OK = 200000  # OK
API_STATUS_CODE_OK_PART = 206000  # 部分结果OK，用于stream
API_STATUS_CODE_CLIENT_ERROR = 400000
API_STATUS_CODE_CLIENT_ERROR_FORMAT = 400001  # 请求数据格式错误
API_STATUS_CODE_CLIENT_ERROR_CONFIG = 400002  # 请求数据配置不支持
API_STATUS_CODE_SERVER_ERROR = 500000
API_STATUS_CODE_SERVER_ERROR_RUNNING = 500001  # 服务器运行中出错

parser = argparse.ArgumentParser(description='ASRT gRPC Protocol API Service')
parser.add_argument('--listen', default='0.0.0.0', type=str, help='the network to listen')
parser.add_argument('--port', default='20002', type=str, help='the port to listen')
args = parser.parse_args()

AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1
# 默认输出的拼音的表示大小是1428，即1427个拼音+1个空白块
OUTPUT_SIZE = 1428
sm251bn = SpeechModel251BN(
    input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS),
    output_size=OUTPUT_SIZE
)
feat = Spectrogram()
ms = ModelSpeech(sm251bn, feat, max_label_length=64)
ms.load_model('save_models/' + sm251bn.get_model_name() + '.model.h5')

ml = ModelLanguage('model_language')
ml.load_model()

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ApiService(AsrtGrpcServiceServicer):
    """
    继承AsrtGrpcServiceServicer,实现hello方法
    """

    def __init__(self):
        pass

    def Speech(self, request, context):
        """
        具体实现Speech的方法, 并按照pb的返回对象构造SpeechResponse返回
        :param request:
        :param context:
        :return:
        """
        wav_data = request.wav_data
        wav_samples = decode_wav_bytes(samples_data=wav_data.samples,
                                       channels=wav_data.channels, byte_width=wav_data.byte_width)
        result = ms.recognize_speech(wav_samples, wav_data.sample_rate)
        print("语音识别声学模型结果:", result)
        return SpeechResponse(status_code=API_STATUS_CODE_OK, status_message='',
                              result_data=result)

    def Language(self, request, context):
        """
        具体实现Language的方法, 并按照pb的返回对象构造TextResponse返回
        :param request:
        :param context:
        :return:
        """
        print('Language收到了请求:', request)
        result = ml.pinyin_to_text(list(request.pinyins))
        print('Language结果:', result)
        return TextResponse(status_code=API_STATUS_CODE_OK, status_message='',
                            text_result=result)

    def All(self, request, context):
        """
        具体实现All的方法, 并按照pb的返回对象构造TextResponse返回
        :param request:
        :param context:
        :return:
        """
        wav_data = request.wav_data
        wav_samples = decode_wav_bytes(samples_data=wav_data.samples,
                                       channels=wav_data.channels, byte_width=wav_data.byte_width)
        result_speech = ms.recognize_speech(wav_samples, wav_data.sample_rate)
        result = ml.pinyin_to_text(result_speech)
        print("语音识别结果:", result)
        return TextResponse(status_code=API_STATUS_CODE_OK, status_message='',
                            text_result=result)

    def Stream(self, request_iterator, context):
        """
        具体实现Stream的方法, 并按照pb的返回对象构造TextResponse返回
        :param request_iterator:
        :param context:
        :return:
        """
        result = list()
        tmp_result_last = list()
        beam_size = 100

        for request in request_iterator:
            wav_data = request.wav_data
            wav_samples = decode_wav_bytes(samples_data=wav_data.samples,
                                           channels=wav_data.channels,
                                           byte_width=wav_data.byte_width)
            result_speech = ms.recognize_speech(wav_samples, wav_data.sample_rate)

            for item_pinyin in result_speech:
                tmp_result = ml.pinyin_stream_decode(tmp_result_last, item_pinyin, beam_size)
                if len(tmp_result) == 0 and len(tmp_result_last) > 0:
                    result.append(tmp_result_last[0][0])
                    print("流式语音识别结果：", ''.join(result))
                    yield TextResponse(status_code=API_STATUS_CODE_OK, status_message='',
                                       text_result=''.join(result))
                    result = list()

                    tmp_result = ml.pinyin_stream_decode([], item_pinyin, beam_size)
                tmp_result_last = tmp_result
                yield TextResponse(status_code=API_STATUS_CODE_OK_PART, status_message='',
                                   text_result=''.join(tmp_result[0][0]))

        if len(tmp_result_last) > 0:
            result.append(tmp_result_last[0][0])
            print("流式语音识别结果：", ''.join(result))
            yield TextResponse(status_code=API_STATUS_CODE_OK, status_message='',
                               text_result=''.join(result))


def run(host, port):
    """
    gRPC API服务启动
    :return:
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_AsrtGrpcServiceServicer_to_server(ApiService(), server)
    server.add_insecure_port(''.join([host, ':', port]))
    server.start()
    print("start service...")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    run(host=args.listen, port=args.port)
