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
语音识别API的HTTP服务器程序
"""

import http.server
import urllib
import socket
import keras
from SpeechModel251 import ModelSpeech
from LanguageModel import ModelLanguage


datapath = './'
modelpath = 'model_speech/'
ms = ModelSpeech(datapath)
ms.LoadModel(modelpath + 'm251/speech_model251_e_0_step_12000.h5')

ml = ModelLanguage('model_language')
ml.LoadModel()


class ASRTHTTPHandle(http.server.BaseHTTPRequestHandler):  
    def setup(self):
        self.request.settimeout(10)
        http.server.BaseHTTPRequestHandler.setup(self)

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):  
        buf = 'ASRT_SpeechRecognition API'  
        self.protocal_version = 'HTTP/1.1'   
        
        self._set_response()
        
        buf = bytes(buf,encoding="utf-8")
        self.wfile.write(buf) 

    def do_POST(self):  
        '''
        处理通过POST方式传递过来并接收的语音数据
        通过语音模型和语言模型计算得到语音识别结果并返回
        '''
        path = self.path  
        print(path)  
        #获取post提交的数据  
        datas = self.rfile.read(int(self.headers['content-length']))  
        #datas = urllib.unquote(datas).decode("utf-8", 'ignore') 
        datas = datas.decode('utf-8')
        datas_split = datas.split('&')
        token = ''
        fs = 0
        wavs = []
        
        for line in datas_split:
            [key, value]=line.split('=')
            if('wavs' == key and '' != value):
                wavs.append(int(value))
            elif('fs' == key):
                fs = int(value)
            elif('token' == key ):
                token = value
            #elif('type' == key):
            #    type = value
            else:
                print(key, value)
            
        if(token != 'qwertasd'):
            buf = '403'
            print(buf)
            buf = bytes(buf,encoding="utf-8")
            self.wfile.write(buf)  
            return
        
        if(len(wavs)>0):
            r = self.recognize([wavs], fs)
        else:
            r = ''
        
        if(token == 'qwertasd'):
            buf = r
        else:
            buf = '403'
        
        self._set_response()
        
        print(buf)
        buf = bytes(buf,encoding="utf-8")
        self.wfile.write(buf)  
        
    def recognize(self, wavs, fs):
        r=''
        try:
            r_speech = ms.RecognizeSpeech(wavs, fs)
            print(r_speech)
            str_pinyin = r_speech
            r = ml.SpeechToText(str_pinyin)
        except Exception as ex:
            r=''
            print('[*Message] Server raise a bug. ', ex)
        return r
        pass
    
    def recognize_from_file(self, filename):
        pass


class HTTPServerV6(http.server.HTTPServer):
    address_family = socket.AF_INET6


def start_server(ip, port):  
    if(':' in ip):
        http_server = HTTPServerV6((ip, port), ASRTHTTPHandle)
    else:
        http_server = http.server.HTTPServer((ip, int(port)), ASRTHTTPHandle)
    
    print('服务器已开启')
    
    try:
        http_server.serve_forever() #设置一直监听并接收请求  
    except KeyboardInterrupt:
        pass
    http_server.server_close()
    print('HTTP server closed')


if __name__ == '__main__':
    start_server('', 20000) # For IPv4 Network Only
    #start_server('::', 20000) # For IPv6 Network
