# ASRT: A Deep-Learning-Based Chinese Speech Recognition System
ASRT是一个基于深度学习的中文语音识别系统，如果您觉得喜欢，请点一个 **"Star"** 吧~

[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) 
[![TensorFlow Version](https://img.shields.io/badge/Tensorflow-1.15+-blue.svg)](https://www.tensorflow.org/) 
[![Python Version](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/) 

**ReadMe Language** | 中文版 | [English](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/README_EN.md) |

[**ASRT项目主页**](https://asrt.ailemon.net/) | 
[**发布版下载**](https://asrt.ailemon.net/download) | 
[**查看本项目的Wiki文档**](https://asrt.ailemon.net/docs/) | 
[**实用效果体验Demo**](https://asrt.ailemon.net/demo) | 
[**打赏作者**](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki/donate)

如果程序运行期间或使用中有什么问题，可以及时在issue中提出来，我将尽快做出答复。本项目作者交流QQ群：**894112051**

提问前请仔细查看[项目文档](https://asrt.ailemon.net/docs/)、 
[常见问题](https://asrt.ailemon.net/docs/issues)
以及[Issues](https://github.com/nl8590687/ASRT_SpeechRecognition/issues) 避免重复提问

以下问题AI柠檬博主和群友可能会拒绝回答，包括但不限于：
* 询问已经写在 **ASRT语音识别项目文档** 和 **Issues** 上解决过的已知重复问题。
* 找不到重点、不知所云的提问内容，但是不给出任何其他信息。
* 跟ASRT项目没有直接相关的问题
* “伸手党”类的问题

```
请注意，开发者并没有义务回复您的问题，也没用义务免费给你打工，您应该具备基本的提问技巧，并善用搜索引擎，
每个人的时间都是宝贵的。
```
有关AI柠檬ASRT语音项目的相关信息亦可使用[AI柠檬站内搜索引擎](https://s.ailemon.net/)进行相关信息的搜索。

## ASRT相关资料 

ASRT的原理请查看本文：
* [ASRT：一个中文语音识别系统](https://blog.ailemon.net/2018/08/29/asrt-a-chinese-speech-recognition-system/)

ASRT训练和部署教程请看：
* [教你如何使用ASRT训练中文语音识别模型](<https://blog.ailemon.net/2020/08/20/teach-you-how-use-asrt-train-chinese-asr-model/>)
* [教你如何使用ASRT部署中文语音识别API服务器](<https://blog.ailemon.net/2020/08/27/teach-you-how-use-asrt-deploy-chinese-asr-api-server/>)

关于经常被问到的统计语言模型原理的问题，请看：

* [统计语言模型：从中文拼音到文本](https://blog.ailemon.net/2017/04/27/statistical-language-model-chinese-pinyin-to-words/)
* [统计N元语言模型生成算法：简单中文词频统计](https://blog.ailemon.net/2017/02/20/simple-words-frequency-statistic-without-segmentation-algorithm/)

关于CTC的问题请看：

* [[翻译]使用CTC进行序列建模](<https://blog.ailemon.net/2019/07/18/sequence-modeling-with-ctc/>)

更多内容请访问作者的博客：[AI柠檬博客](https://blog.ailemon.net/)

或使用[AI柠檬站内搜索引擎](https://s.ailemon.net/)进行相关信息的搜索

## Introduction 简介

本项目使用tensorFlow.keras基于深度卷积神经网络和长短时记忆神经网络、注意力机制以及CTC实现。

This project uses tensorFlow.keras based on deep convolutional neural network and long-short memory neural network, attention mechanism and CTC to implement.

* **操作步骤**

首先通过Git将本项目克隆到您的计算机上，然后下载本项目训练所需要的数据集，下载链接详见[文档末尾部分](https://github.com/nl8590687/ASRT_SpeechRecognition#data-sets-%E6%95%B0%E6%8D%AE%E9%9B%86)。
```shell
$ git clone https://github.com/nl8590687/ASRT_SpeechRecognition.git
```

或者您也可以通过 "Fork" 按钮，将本项目Copy一份副本，然后通过您自己的SSH密钥克隆到本地。

通过git克隆仓库以后，进入项目根目录；并创建一个存储数据的子目录， 例如 `dataset/` (可使用软链接代替)，然后将下载好的数据集直接解压进去

注意，当前版本中，在配置文件里，默认添加了Thchs30和ST-CMDS两个数据集，如果不需要请自行删除。如果要使用其他数据集需要自行添加数据配置，并提前使用ASRT支持的标准格式整理数据。

```shell
$ cd ASRT_SpeechRecognition

$ mkdir dataset

$ tar zxf <数据集压缩文件名> -C dataset/ 
```

然后需要将datalist目录下的文件全部拷贝到 `dataset/` 目录下，也就是将其跟数据集放在一起。
```shell
$ cp -rf datalist/* dataset/
```

目前可用的模型有24、25和251

运行本项目之前，请安装必要的[Python3版依赖库](https://github.com/nl8590687/ASRT_SpeechRecognition#python-import)

本项目开始训练请执行：
```shell
$ python3 train_speech_model.py
```
本项目开始测试请执行：
```shell
$ python3 evaluate_speech_model.py
```
测试之前，请确保代码中填写的模型文件路径存在。

ASRT API服务器启动请执行：
```shell
$ python3 asrserver.py
```

请注意，开启API服务器之后，需要使用本ASRT项目对应的客户端软件来进行语音识别，详见Wiki文档[ASRT客户端Demo](https://asrt.ailemon.net/docs/client-demo)。

如果要训练和使用非251版模型，请在代码中 `import speech_model_zoo` 的相应位置做修改。

## Model 模型

### Speech Model 语音模型

CNN/LSTM/GRU + CTC

其中，输入的音频的最大时间长度为16秒，输出为对应的汉语拼音序列

* 关于下载已经训练好的模型的问题

已经训练好的模型包含在发布版服务端程序压缩包里面，发布版成品服务端程序可以在此下载：[ASRT下载页面](https://asrt.ailemon.net/download)。

Github本仓库下[Releases](https://github.com/nl8590687/ASRT_SpeechRecognition/releases)页面里面还包括各个不同版本的介绍信息，每个版本下方的zip压缩包也是包含已经训练好的模型的发布版服务端程序压缩包。

### Language Model 语言模型

基于概率图的最大熵隐马尔可夫模型

输入为汉语拼音序列，输出为对应的汉字文本

## About Accuracy 关于准确率

当前，最好的模型在测试集上基本能达到80%的汉语拼音正确率

不过由于目前国际和国内的部分团队能做到98%，所以正确率仍有待于进一步提高

## Python Dependency Library
Python的依赖库

* tensorFlow (1.15 - 2.x)
* numpy
* wave
* matplotlib
* math
* scipy
* requests

不会安装环境的同学请直接运行以下命令(前提是有GPU且已经安装好 CUDA 11.2 和 cudnn 8.1)：

```shell
$ pip install -r requirements.txt
```

[程序运行依赖环境详细说明](https://asrt.ailemon.net/docs/dependent-environment)

## Data Sets 数据集

[几个最新免费开源的中文语音数据集](https://blog.ailemon.net/2018/11/21/free-open-source-chinese-speech-datasets/)

* **清华大学THCHS30中文语音数据集**

  data_thchs30.tgz 
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/18/data_thchs30.tgz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/18/data_thchs30.tgz>)

  test-noise.tgz 
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/18/test-noise.tgz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/18/test-noise.tgz>)

  resource.tgz 
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/18/resource.tgz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/18/resource.tgz>)

* **Free ST Chinese Mandarin Corpus** 

  ST-CMDS-20170001_1-OS.tar.gz 
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/38/ST-CMDS-20170001_1-OS.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz>)

* **AIShell-1 开源版数据集** 

  data_aishell.tgz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/33/data_aishell.tgz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/33/data_aishell.tgz>)

  注：数据集解压方法

  ```
  $ tar xzf data_aishell.tgz
  $ cd data_aishell/wav
  $ for tar in *.tar.gz;  do tar xvf $tar; done
  ```

* **Primewords Chinese Corpus Set 1** 

  primewords_md_2018_set1.tar.gz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/47/primewords_md_2018_set1.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/47/primewords_md_2018_set1.tar.gz>)

* **aidatatang_200zh**

   aidatatang_200zh.tgz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/62/aidatatang_200zh.tgz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/62/aidatatang_200zh.tgz>)

* **MagicData**

  train_set.tar.gz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/68/train_set.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/68/train_set.tar.gz>)

  dev_set.tar.gz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/68/dev_set.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/68/dev_set.tar.gz>)

  test_set.tar.gz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/68/test_set.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/68/test_set.tar.gz>)

  metadata.tar.gz
[OpenSLR国内镜像](<http://openslr.magicdatatech.com/resources/68/metadata.tar.gz>)
[OpenSLR国外镜像](<http://www.openslr.org/resources/68/metadata.tar.gz>)

特别鸣谢！感谢前辈们的公开语音数据集

如果提供的数据集链接无法打开和下载，请点击该链接 [OpenSLR](http://www.openslr.org)

## License 开源许可协议

[GPL v3.0](LICENSE) © [nl8590687](https://github.com/nl8590687) 作者：[AI柠檬](https://www.ailemon.net/)

## Contributors 贡献者们

[@zw76859420](https://github.com/zw76859420) 
@madeirak @ZJUGuoShuai @williamchenwl

@nl8590687 (repo owner)
