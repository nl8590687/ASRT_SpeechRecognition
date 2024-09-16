![](assets/asrt_title_header_en.png)

[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) 
[![Stars](https://img.shields.io/github/stars/nl8590687/ASRT_SpeechRecognition)](https://github.com/nl8590687/ASRT_SpeechRecognition) 
[![TensorFlow Version](https://img.shields.io/badge/Tensorflow-2.5+-blue.svg)](https://www.tensorflow.org/) 
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5808434.svg)](https://doi.org/10.5281/zenodo.5808434)

ASRT is A Deep-Learning-Based Chinese Speech Recognition System. If you like this project, please **star** it. 

**ReadMe Language** | [中文版](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/README.md) | English |

[**ASRT Project Home Page**](https://asrt.ailemon.net/) | 
[**Released Download**](https://wiki.ailemon.net/docs/asrt-doc/download) | 
[**View this project's wiki document (Chinese)**](https://wiki.ailemon.net/docs/asrt-doc) | 
[**Experience Demo**](https://asrt.ailemon.net/demo) | 
[**Donate**](https://wiki.ailemon.net/docs/asrt-doc/asrt-doc-1deo9u61unti9)

If you have any questions in your works with this project, welcome to put up issues in this repo and I will response as soon as possible. 

You can check the [FAQ Page (Chinese)](https://wiki.ailemon.net/docs/asrt-doc/asrt-doc-1deoeud494h4f) first before asking questions to avoid repeating questions.

If there is any abnormality when the program is running, please send a complete screenshot when asking questions, and indicate the CPU architecture, GPU model, operating system, Python, TensorFlow and CUDA versions used, and whether any code has been modified or data sets have been added or deleted, etc. .

## Introduction

This project uses tensorFlow.keras based on deep convolutional neural network and long-short memory neural network, attention mechanism and CTC to implement. 

## Minimum requirements for training
### Hardware
* CPU: 4 Core (x86_64, amd64) +
* RAM: 16 GB +
* GPU: NVIDIA, Graph Memory 11GB+ (>1080ti)
* 硬盘: 500 GB HDD(or SSD)

### Software
* Linux: Ubuntu 20.04+ / CentOS 7+ (train & predict) or Windows: 10/11 (only to predict)
* Python: 3.9 - 3.11 and later
* TensorFlow: 2.5 - 2.11 and later

## Quick Start
Take the operation under the Linux system as an example:

First, clone the project to your computer through Git, and then download the data sets needed for the training of this project. For the download links, please refer to [End of Document](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/README_EN.md#data-sets)
```shell
$ git clone https://github.com/nl8590687/ASRT_SpeechRecognition.git
```

Or you can use the "Fork" button to copy a copy of the project and then clone it locally with your own SSH key.

After cloning the repository via git, go to the project root directory; create a subdirectory `/data/speech_data` (you can use a soft link instead) for datasets, and then extract the downloaded datasets directly into it.

```shell
$ cd ASRT_SpeechRecognition

$ mkdir /data/speech_data

$ tar zxf <dataset zip files name> -C /data/speech_data/ 
```

Note that in the current version, in the configuration file, six data sets, Thchs30, ST-CMDS, Primewords, aishell-1, aidatatang200, MagicData, are added by default, please delete them if you don’t need them. If you want to use other data sets, you need to add data configuration yourself, and use the standard format supported by ASRT to organize the data in advance.

To download pinyin syllable list files for default dataset:
```shell
$ python download_default_datalist.py
```

Currently available models are 24, 25, 251 and 251bn

Before running this project, please install the necessary [Python3 version dependent library](https://github.com/nl8590687/ASRT_SpeechRecognition#python-import)

To start training this project, please execute:
```shell
$ python3 train_speech_model.py
```
To start the test of this project, please execute:
```shell
$ python3 evaluate_speech_model.py
```
Before testing, make sure the model file path filled in the code files exists.

To predict one wave audio file for speech recognition：
```shell
$ python3 predict_speech_file.py
```

To startup ASRT API Server with HTTP protocol please execute:
```shell
$ python3 asrserver_http.py
```

Please note that after opening the API server, you need to use the client software corresponding to this ASRT project for voice recognition. For details, see the Wiki documentation to [download ASRT Client SDK & Demo](https://wiki.ailemon.net/docs/asrt-doc/download).


To test whether it is successful or not that calls api service interface with HTTP protocol:
```shell
$ python3 client_http.py
```

To startup ASRT API Server with GRPC protocol please execute:
```shell
$ python3 asrserver_grpc.py
```

To test whether it is successful or not that calls api service interface with GRPC protocol:
```shell
$ python3 client_grpc.py
```

If you want to train and use other model(not Model 251bn), make changes in the corresponding position of the `from speech_model.xxx import xxx` in the code files.

If there is any problem during the execution of the program or during use, it can be promptly put forward in the issue, and I will reply as soon as possible.

Deploy ASRT by docker：
```shell
$ docker pull ailemondocker/asrt_service:1.3.0
$ docker run --rm -it -p 20001:20001 -p 20002:20002 --name asrt-server -d ailemondocker/asrt_service:1.3.0
```
It will start a api server for recognition rather than training.

## Model

### Speech Model

DCNN + CTC

The maximum length of the input audio is 16 seconds, and the output is the corresponding Chinese pinyin sequence. 

* Questions about downloading trained models

The released finished software that includes trained model weights can be downloaded from [ASRT download page](https://wiki.ailemon.net/docs/asrt-doc/download). 

Github [Releases](https://github.com/nl8590687/ASRT_SpeechRecognition/releases) page includes the archives of the various versions of the software released and it's introduction. Under each version module, there is a zip file that includes trained model weights files. 

### Language Model 

Maximum Entropy Hidden Markov Model Based on Probability Graph. 

The input is a Chinese pinyin sequence, and the output is the corresponding Chinese character text. 

## About Accuracy

At present, the best model can basically reach 85% of Pinyin correct rate on the test set. 

## Python Dependency Library

* tensorFlow (2.5-2.11+)
* numpy
* wave
* matplotlib
* math
* scipy
* requests
* flask
* waitress
* grpcio / grpcio-tools / protobuf

If you have trouble when install those packages, please run the following script to do it as long as you have a GPU and python 3.9, CUDA 11.2 and cudnn 8.1 have been installed：

```shell
$ pip install -r requirements.txt
```

[Dependent Environment Details and Hardware Requirement](https://wiki.ailemon.net/docs/asrt-doc/asrt-doc-1deobk7bmlgd6)

## ASRT Client SDK for Calling Speech Recognition API

ASRT provides the abilities to import client SDKs for several platform and programing language for client develop speech recognition features , which work by RPC. Please refer ASRT project documents for detail.

|Client Platform|Project Repos Link|
|-|-|
|Windows Client SDK & Demo|[ASRT_SDK_WinClient](https://github.com/nl8590687/ASRT_SDK_WinClient)|
|Python3 Client SDK & Demo (Any Platform)|[ASRT_SDK_Python3](https://github.com/nl8590687/ASRT_SDK_Python3)|
|Golang Client SDK & Demo|[asrt-sdk-go](https://github.com/nl8590687/asrt-sdk-go)|
|Java Client SDK & Demo|[ASRT_SDK_Java](https://github.com/nl8590687/ASRT_SDK_Java)|

## Data Sets 

For full content please refer: [Some free Chinese speech datasets (Chinese)](https://blog.ailemon.net/2018/11/21/free-open-source-chinese-speech-datasets/)

|Dataset|Time|Size|Download (CN Mirrors)|Download (Source)|
|-|-|-|-|-|
|THCHS30|40h|6.01G|[data_thchs30.tgz](<http://openslr.magicdatatech.com/resources/18/data_thchs30.tgz>)|[data_thchs30.tgz](<http://www.openslr.org/resources/18/data_thchs30.tgz>)|
|ST-CMDS|100h|7.67G|[ST-CMDS-20170001_1-OS.tar.gz](<http://openslr.magicdatatech.com/resources/38/ST-CMDS-20170001_1-OS.tar.gz>)|[ST-CMDS-20170001_1-OS.tar.gz](<http://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz>)|
|AIShell-1|178h|14.51G|[data_aishell.tgz](<http://openslr.magicdatatech.com/resources/33/data_aishell.tgz>)|[data_aishell.tgz](<http://www.openslr.org/resources/33/data_aishell.tgz>)|
|Primewords|100h|8.44G|[primewords_md_2018_set1.tar.gz](<http://openslr.magicdatatech.com/resources/47/primewords_md_2018_set1.tar.gz>)|[primewords_md_2018_set1.tar.gz](<http://www.openslr.org/resources/47/primewords_md_2018_set1.tar.gz>)|
|MagicData|755h|52G/1.0G/2.2G| [train_set.tar.gz](<http://openslr.magicdatatech.com/resources/68/train_set.tar.gz>) / [dev_set.tar.gz](<http://openslr.magicdatatech.com/resources/68/dev_set.tar.gz>) / [test_set.tar.gz](<http://openslr.magicdatatech.com/resources/68/test_set.tar.gz>)|[train_set.tar.gz](<http://www.openslr.org/resources/68/train_set.tar.gz>) / [dev_set.tar.gz](<http://www.openslr.org/resources/68/dev_set.tar.gz>) / [test_set.tar.gz](<http://www.openslr.org/resources/68/test_set.tar.gz>)|


  Note：The way to unzip AISHELL-1 dataset

  ```
  $ tar xzf data_aishell.tgz
  $ cd data_aishell/wav
  $ for tar in *.tar.gz;  do tar xvf $tar; done
  ```

Special thanks! Thanks to the predecessors' public voice data set. 

If the provided dataset link cannot be opened and downloaded, click this link [OpenSLR](http://www.openslr.org)

## ASRT Docuemnts

* [ASRT project's Wiki document](https://wiki.ailemon.net/docs/asrt-doc)

A post about ASRT's introduction 
* [ASRT: Chinese Speech Recognition System (Chinese)](https://blog.ailemon.net/2018/08/29/asrt-a-chinese-speech-recognition-system/)

About how to use ASRT to train and deploy：
* [Teach you how to use ASRT to train Chinese ASR model (Chinese)](<https://blog.ailemon.net/2020/08/20/teach-you-how-use-asrt-train-chinese-asr-model/>)
* [Teach you how to use ASRT to deploy Chinese ASR API Server (Chinese)](<https://blog.ailemon.net/2020/08/27/teach-you-how-use-asrt-deploy-chinese-asr-api-server/>)

For questions about the principles of the statistical language model that are often asked, see: 
* [Simple Chinese word frequency statistics to generate N-gram language model (Chinese)](https://blog.ailemon.net/2017/02/20/simple-words-frequency-statistic-without-segmentation-algorithm/)
* [Statistical Language Model: Chinese Pinyin to Words (Chinese)](https://blog.ailemon.net/2017/04/27/statistical-language-model-chinese-pinyin-to-words/)

For questions about CTC, see: 

* [[Translation] Sequence Modeling with CTC (Chinese)](<https://blog.ailemon.net/2019/07/18/sequence-modeling-with-ctc/>)

For more infomation please refer to author's blog website: [AILemon Blog](https://blog.ailemon.net/) (Chinese)

## License

[GPL v3.0](LICENSE) © [nl8590687](https://github.com/nl8590687) Author: [ailemon](https://www.ailemon.net/)

## Cite this project

[DOI: 10.5281/zenodo.5808434](https://doi.org/10.5281/zenodo.5808434)

## Contributors

[Contributors Page](https://github.com/nl8590687/ASRT_SpeechRecognition/graphs/contributors)

@nl8590687 (repo owner)
