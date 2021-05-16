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
用于训练语音识别系统语音模型的程序

"""
import platform as plat
import os

import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session


from SpeechModel251 import ModelSpeech, ModelName

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#进行配置，使用95%的GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
#config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


datapath = ''
modelpath = 'model_speech'


if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
	os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉
	os.makedirs(modelpath + '/m' + ModelName)

system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
if(system_type == 'Windows'):
	datapath = 'D:\\SpeechData'
	modelpath = modelpath + '\\'
elif(system_type == 'Linux'):
	datapath = 'dataset'
	modelpath = modelpath + '/'
else:
	print('*[Message] Unknown System\n')
	datapath = 'dataset'
	modelpath = modelpath + '/'

ms = ModelSpeech(datapath)

#ms.LoadModel(modelpath + 'speech_model251_e_0_step_327500.h5')
ms.TrainModel(datapath, epoch = 50, batch_size = 16, save_step = 500)


