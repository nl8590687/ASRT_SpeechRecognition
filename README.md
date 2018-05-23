# ASRT_SpeechRecognition
基于深度学习的语音识别系统

## Introduction 简介

本项目使用Keras、TensorFlow基于长短时记忆神经网络和卷积神经网络以及CTC进行制作。

This project uses keras, TensorFlow based on LSTM, CNN and CTC to implement. 

[查看本项目的Wiki页面](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki) (正在完善中)

本项目目前已经可以正常进行训练了。

通过git克隆仓库以后，需要将datalist目录下的文件全部拷贝到dataset目录下，也就是将其跟数据集放在一起。
```shell
$ cp -rf datalist/* dataset/
```

目前可用的模型有22和24

本项目开始训练请执行：
```shell
$ python3 train_mspeech.py
```
本项目开始测试请执行：
```shell
$ python3 test_mspeech.py
```
测试之前，请确保代码中填写的模型文件路径存在。

ASRT API服务器启动请执行：
```shell
$ python3 asrserver.py
```

如果要训练和使用模型24，请在代码中 `import SpeechModel` 的相应位置做修改。

如果程序运行期间或使用中有什么问题，可以及时在issue中提出来，我将尽快做出答复。

提问前可以先 [查看常见问题](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki/issues) 

## Model 模型

### Speech Model 语音模型

CNN + LSTM/GRU + CTC

### Language Model 语言模型

基于概率图的马尔可夫模型

## About Accuracy 关于准确率

当前，speech_model22的准确率在GPU上训练了120+小时（大约50个epoch），在测试集上基本能达到70+%的汉语拼音正确率

不过由于目前国际和国内的部分团队能做到97%，所以正确率仍有待于进一步提高

* 目前可知的可以继续提高准确率的一个方案就是纠正数据集标注错误，尤其是ST-CMDS里面关于syllable文件中拼音的错误，这里面有一定比例的错误标注，如果走过路过的各位有意愿帮助纠正数据标注错误的，可以提交Pull Request，并且将登上本仓库的贡献者名单。

样例：`不是： bu4 shi4 -> bu2 shi4` `一个：yi1 ge4 -> yi2 ge4` `了解：le5 jie3 -> liao3 jie3`

* 已订正部分：

ST-CMDS

train:  20170001P00001A    20170001P00001I    20170001P00002A

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

## Contributors 贡献者们
@ZJUGuoShuai @williamchenwl

@nl8590687 (repo owner)