#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
语音识别API的HTTP服务器程序

"""
import http.server
import urllib
import keras
from SpeechModel24 import ModelSpeech
from LanguageModel import ModelLanguage

datapath = 'data/'
modelpath = 'model_speech/'
ms = ModelSpeech(datapath)
ms.LoadModel(modelpath + 'm24/speech_model24_e_0_step_411000.model')

ml = ModelLanguage('model_language')
ml.LoadModel()

class TestHTTPHandle(http.server.BaseHTTPRequestHandler):  
	
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
		#type = 'wavfilebytes' # wavfilebytes or python-list
		
		for line in datas_split:
			[key, value]=line.split('=')
			if('wavs' == key and '' != value):
				wavs.append(int(value))
			elif('fs' == key):
				fs = int(value)
			elif('token' == key ):
				token = value
			#elif('type' == key):
			#	type = value
			else:
				print(key, value)
			
		if(token != 'qwertasd'):
			buf = '403'
			print(buf)
			buf = bytes(buf,encoding="utf-8")
			self.wfile.write(buf)  
			return
		
		#if('python-list' == type):
		if(len(wavs)>0):
			r = self.recognize([wavs], fs)
		else:
			r = ''
		#else:
		#	r = self.recognize_from_file('')
		
		if(token == 'qwertasd'):
			#buf = '成功\n'+'wavs:\n'+str(wavs)+'\nfs:\n'+str(fs)
			buf = r
		else:
			buf = '403'
		
		#print(datas)
		
		self._set_response()
		
		#buf = '<!DOCTYPE HTML> \n<html> \n<head>\n<title>Post page</title>\n</head> \n<body>Post Data:%s  <br />Path:%s\n</body>  \n</html>'%(datas,self.path)  
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
		except:
			r=''
			print('[*Message] Server raise a bug. ')
		return r
		pass
	
	def recognize_from_file(self, filename):
		pass
	
def start_server(ip, port):  
	http_server = http.server.HTTPServer((ip, int(port)), TestHTTPHandle)  
	print('服务器已开启')
	
	try:
		http_server.serve_forever() #设置一直监听并接收请求  
	except KeyboardInterrupt:
		pass
	http_server.server_close()
	print('HTTP server closed')
	
if __name__ == '__main__':
	
	start_server('', 20000)
	
	
	
