
import numpy
import math
import tensorflow as tf

def sigmoid(theta,X):
	'''
	激活函数sigmoid
	参数theta和X均为矩阵类型
	返回计算后的函数值
	'''
	return float(1/(1+math.exp(-theta*X)))
	
def nnCostFunction():
	'''
	计算神经网络的代价函数
	返回一个Cost值
	'''
	
def compute(theta,X):
	'''
	正向计算神经网络的函数结果
	返回最终的结果
	'''