# ASRT_SpeechRecognition
基于深度学习的语音识别系统

## Introduction

这里是更新记录日志文件

如果有什么问题，可以在这里直接写出来

## Log
### 2018-03-30
暂时能够正常训练模型了，修复了一大堆bug问题，也在这个过程中学到很多。感谢队友和所有直接或间接帮助过我的各位大大。继续前进！
### 2018-03-28
CTC这一块似乎添加成功了？现在开始debug了。。。
现在也许可以开始训练了
### 2018-03-11
添加了神经网络的CTC层和定义了CTC_loss损失函数，但是现在有些严重的bug，使得模型无法正常编译，一直找不到问题所在......(T_T)
#### 报错 
ValueError: Shapes (?, ?) and (?,) must have the same rank

ValueError: Shapes (?, ?) and (?,) are not compatible

ValueError: Shape (?, ?) must have rank 1
#### --------------------------------
各位走过路过的大神有会的吗？请帮帮忙吧，ヾ(o′▽`o)ノ°°谢谢啦
### 2017-09-08
基本完成除了添加模型之外的其他部分代码
### 2017-08-31
数据处理部分的代码基本完成，现在准备撸模型
### 2017-08-29
准备使用现有的包[python_speech_features](https://github.com/jameslyons/python_speech_features)来实现特征的提取，以及求一阶二阶差分。
### 2017-08-28
开始准备制作语音信号处理方面的功能
### 2017-08-22
准备使用Keras基于LSTM/CNN尝试实现

