# ASRT: A Deep-Learning-Based Chinese Speech Recognition System

[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) 
[![TensorFlow Version](https://img.shields.io/badge/Tensorflow-1.15+-blue.svg)](https://www.tensorflow.org/) 
[![Python Version](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/) 

**ReadMe Language** | [中文版](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/README.md) | English |

[**ASRT Project Home Page**](https://asrt.ailemon.net/) | 
[**Released Download**](https://asrt.ailemon.net/download) | 
[**View this project's wiki document (Chinese)**](https://wiki.ailemon.net/docs/asrt-doc) | 
[**Experience Demo**](https://asrt.ailemon.net/demo) | 
[**Donate**](https://wiki.ailemon.net/docs/asrt-doc/asrt-doc-1deo9u61unti9)

If you have any questions in your works with this project, welcome to put up issues in this repo and I will response as soon as possible. 

You can check the [FAQ Page (Chinese)](https://wiki.ailemon.net/docs/asrt-doc/asrt-doc-1deoeud494h4f) first before asking questions to avoid repeating questions.

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

## Introduction

This project uses tensorFlow.keras based on deep convolutional neural network and long-short memory neural network, attention mechanism and CTC to implement. 

* **Steps**

First, clone the project to your computer through Git, and then download the data sets needed for the training of this project. For the download links, please refer to [End of Document](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/README_EN.md#data-sets)
```shell
$ git clone https://github.com/nl8590687/ASRT_SpeechRecognition.git
```

Or you can use the "Fork" button to copy a copy of the project and then clone it locally with your own SSH key.

After cloning the repository via git, go to the project root directory; create a subdirectory `dataset/` (you can use a soft link instead) for datasets, and then extract the downloaded datasets directly into it.

```shell
$ cd ASRT_SpeechRecognition

$ mkdir dataset

$ tar zxf <dataset zip files name> -C dataset/ 
```

Then, you need to copy all the files in the 'datalist' directory to the dataset directory, that is, put them together with the data set.

Note that in the current version, in the configuration file, two data sets, Thchs30 and ST-CMDS, are added by default, please delete them if you don’t need them. If you want to use other data sets, you need to add data configuration yourself, and use the standard format supported by ASRT to organize the data in advance.

```shell
$ cp -rf datalist/* dataset/
```

Currently available models are 24, 25 and 251

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

ASRT API Server startup please execute:
```shell
$ python3 asrserver.py
```

Please note that after opening the API server, you need to use the client software corresponding to this ASRT project for voice recognition. For details, see the Wiki documentation to [download ASRT Client Demo](https://wiki.ailemon.net/docs/asrt-doc/download).

If you want to train and use other model(not Model 251), make changes in the corresponding position of the `import speech_model_zoo` in the code files.

If there is any problem during the execution of the program or during use, it can be promptly put forward in the issue, and I will reply as soon as possible.

Deploy ASRT by docker：
```shell
$ docker pull ailemondocker/asrt_service:1.1.0
$ docker run --rm -it -p 20000:20000 --name asrt-server -d ailemondocker/asrt_service:1.1.0
```
It will start a api server for recognition rather than training.

## Model

### Speech Model

CNN/LSTM/GRU + CTC

The maximum length of the input audio is 16 seconds, and the output is the corresponding Chinese pinyin sequence. 

* Questions about downloading trained models

The released finished software that includes trained model weights can be downloaded from [ASRT download page](https://wiki.ailemon.net/docs/asrt-doc/download). 

Github [Releases](https://github.com/nl8590687/ASRT_SpeechRecognition/releases) page includes the archives of the various versions of the software released and it's introduction. Under each version module, there is a zip file that includes trained model weights files. 

### Language Model 

Maximum Entropy Hidden Markov Model Based on Probability Graph. 

The input is a Chinese pinyin sequence, and the output is the corresponding Chinese character text. 

## About Accuracy

At present, the best model can basically reach 80% of Pinyin correct rate on the test set. 

However, as the current international and domestic teams can achieve 98%, the accuracy rate still needs to be further improved. 

## Python Dependency Library

* tensorFlow (1.15 - 2.x)
* numpy
* wave
* matplotlib
* math
* scipy
* requests

If you have trouble when install those packages, please run the following script to do it as long as you have a GPU and CUDA 11.2 and cudnn 8.1 have been installed：

```shell
$ pip install -r requirements.txt
```

[Dependent Environment Details and Hardware Requirement](https://wiki.ailemon.net/docs/asrt-doc/asrt-doc-1deobk7bmlgd6)

## Data Sets 

[Some free Chinese speech datasets (Chinese)](https://blog.ailemon.net/2018/11/21/free-open-source-chinese-speech-datasets/)

* **Tsinghua University THCHS30 Chinese voice data set**

  data_thchs30.tgz 
[Download](<http://www.openslr.org/resources/18/data_thchs30.tgz>)

  test-noise.tgz 
[Download](<http://www.openslr.org/resources/18/test-noise.tgz>)

  resource.tgz 
[Download](<http://www.openslr.org/resources/18/resource.tgz>)

* **Free ST Chinese Mandarin Corpus**

  ST-CMDS-20170001_1-OS.tar.gz 
[Download](<http://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz>)

* **AIShell-1 Open Source Dataset** 

  data_aishell.tgz
[Download](<http://www.openslr.org/resources/33/data_aishell.tgz>)

  Note：unzip this dataset

  ```
  $ tar xzf data_aishell.tgz
  $ cd data_aishell/wav
  $ for tar in *.tar.gz;  do tar xvf $tar; done
  ```

* **Primewords Chinese Corpus Set 1** 

  primewords_md_2018_set1.tar.gz
[Download](<http://www.openslr.org/resources/47/primewords_md_2018_set1.tar.gz>)

* **aidatatang_200zh**

  aidatatang_200zh.tgz
[Download](<http://www.openslr.org/resources/62/aidatatang_200zh.tgz>)

* **MagicData**

  train_set.tar.gz
[Download](<http://www.openslr.org/resources/68/train_set.tar.gz>)

  dev_set.tar.gz
[Download](<http://www.openslr.org/resources/68/dev_set.tar.gz>)

  test_set.tar.gz
[Download](<http://www.openslr.org/resources/68/test_set.tar.gz>)

  metadata.tar.gz
[Download](<http://www.openslr.org/resources/68/metadata.tar.gz>)

Special thanks! Thanks to the predecessors' public voice data set. 

If the provided dataset link cannot be opened and downloaded, click this link [OpenSLR](http://www.openslr.org)

## License

[GPL v3.0](LICENSE) © [nl8590687](https://github.com/nl8590687) Author: [ailemon](https://www.ailemon.net/)

## Cite this project

[DOI: 10.5281/zenodo.5808435](https://doi.org/10.5281/zenodo.5808435)

## Contributors

[Contributors Page](https://github.com/nl8590687/ASRT_SpeechRecognition/graphs/contributors)

@nl8590687 (repo owner)
