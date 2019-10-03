#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
获取符号字典列表的程序
'''
import os
import platform as plat

def GetSymbolList(datapath):
	'''
	加载拼音符号列表，用于标记符号
	返回一个列表list类型变量
	'''
	txt_obj=open(os.path.join(datapath, 'dict.txt'),'r',encoding='UTF-8') # 打开文件并读入
	txt_text=txt_obj.read()
	txt_lines=txt_text.split('\n') # 文本分割
	list_symbol=[] # 初始化符号列表
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split('\t')
			list_symbol.append(txt_l[0])
	txt_obj.close()
	list_symbol.append('_')
	#SymbolNum = len(list_symbol)
	return list_symbol
	
def GetSymbolList_trash2(datapath):
	'''
	加载拼音符号列表，用于标记符号
	返回一个列表list类型变量
	'''

	datapath_ = datapath.strip('dataset')
	txt_obj=open(os.path.join(datapath_,  'dict.txt'),'r',encoding='UTF-8') # 打开文件并读入
	txt_text=txt_obj.read()        
	txt_lines=txt_text.split('\n') # 文本分割    
	list_symbol=[] # 初始化符号列表
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split('\t')						
			list_symbol.append(txt_l[0])            
	txt_obj.close()
	list_symbol.append('_')
	#SymbolNum = len(list_symbol)
	return list_symbol


if(__name__ == '__main__'):
	GetSymbolList('E:\\abc\\')