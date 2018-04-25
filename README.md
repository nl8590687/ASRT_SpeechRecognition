# ASRT_SpeechRecognition
基于深度学习的语音识别系统

## Introduction 简介

本项目使用Keras、TensorFlow基于长短时记忆神经网络和卷积神经网络以及CTC进行制作。

This project uses keras, TensorFlow based on LSTM, CNN and CTC to implement. 

本项目目前已经可以正常进行训练了。

本项目运行请执行：
```shell
$ python3 SpeechModel22.py
```

## Model 模型

### Speech Model 语音模型

CNN + LSTM + CTC

### Language Model 语言模型

基于概率图的马尔可夫模型

## Python Import
Python的依赖库

* python_speech_features
* TensorFlow
* Keras
* Numpy
* wave
* matplotlib
* math
* Scipy
* h5py

## Data Sets 数据集
* 清华大学THCHS30中文语音数据集

data_thchs30.tgz 
<http://cn-mirror.openslr.org/resources/18/data_thchs30.tgz>

test-noise.tgz 
<http://cn-mirror.openslr.org/resources/18/test-noise.tgz>

resource.tgz 
<http://cn-mirror.openslr.org/resources/18/resource.tgz>

* Free ST Chinese Mandarin Corpus

ST-CMDS-20170001_1-OS.tar.gz 
<http://cn-mirror.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz>

特别鸣谢！感谢前辈们的公开语音数据集

## Log
日志

链接：[进展日志](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/log.md)
