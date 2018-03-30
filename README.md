# ASRT_SpeechRecognition
基于深度学习的语音识别系统

## Introduction 简介

本项目使用Keras、TensorFlow基于长短时记忆神经网络和卷积神经网络以及CTC进行制作。

This project uses keras, TensorFlow based on LSTM, CNN and CTC to implement. 

本项目目前已经可以进行训练了，不过训练时有时候会出现梯度爆炸的问题。

本项目运行请执行：
```shell
$ python3 SpeechModel.py
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
清华大学THCHS30中文语音数据集

wav <http://data.cslt.org/thchs30/zip/wav.tgz>

doc <http://data.cslt.org/thchs30/zip/doc.tgz>

lm <http://data.cslt.org/thchs30/zip/lm.tgz>

特别鸣谢！感谢前辈们的公开语音数据集

## Log
日志

链接：[进展日志](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/log.md)
