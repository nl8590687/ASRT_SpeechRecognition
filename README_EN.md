# A Deep-Learning-Based Chinese Speech Recognition System

[![GPL-3.0 Licensed](https://img.shields.io/badge/License-GPL3.0-blue.svg?style=flat)](https://opensource.org/licenses/GPL-3.0) [![TensorFlow Version](https://img.shields.io/badge/Tensorflow-1.4+-blue.svg)](https://www.tensorflow.org/) [![Keras Version](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) [![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/) 

**ReadMe Language** | [中文版](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/README.md) | English |

[**View this project's wiki document (Chinese)**](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki)

If you have any questions in your works with this project, welcome to put up issues in this repo and I will response as soon as possible. 

You can check the [FAQ Page (Chinese)](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki/issues) first before asking questions to avoid repeating questions.

A post about ASRT's introduction 
* [ASRT: Chinese Speech Recognition System (Chinese)](https://blog.ailemon.me/2018/08/29/asrt-a-chinese-speech-recognition-system/)

For questions about the principles of the statistical language model that are often asked, see: 
* [Simple word frequency statistics without Chinese word segmentation algorithm (Chinese)](https://blog.ailemon.me/2017/02/20/simple-words-frequency-statistic-without-segmentation-algorithm/)
* [Statistical Language Model: Chinese Pinyin to Words (Chinese)](https://blog.ailemon.me/2017/04/27/statistical-language-model-chinese-pinyin-to-words/)

## Introduction

This project uses Keras, TensorFlow based on deep convolutional neural network and long-short memory neural network, attention mechanism and CTC to implement. 

* **Steps**

First, clone the project to your computer through Git, and then download the data sets needed for the training of this project. For the download links, please refer to [End of Document](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/README_EN.md#data-sets)
```shell
$ git clone https://github.com/nl8590687/ASRT_SpeechRecognition.git
```

Or you can use the "Fork" button to copy a copy of the project and then clone it locally with your own SSH key.

After cloning the repository via git, go to the project root directory; create a subdirectory `dataset/` (you can use a soft link instead), and then extract the downloaded datasets directly into it.

```shell
$ cd ASRT_SpeechRecognition

$ mkdir dataset

$ tar zxf <dataset zip files name> -C dataset/ 
```

Then, you need to copy all the files in the 'datalist' directory to the dataset directory, that is, put them together with the data set.

```shell
$ cp -rf datalist/* dataset/
```

Currently available models are 24, 25 and 251

Before running this project, please install the necessary [Python3 version dependent library](https://github.com/nl8590687/ASRT_SpeechRecognition#python-import)

To start training this project, please execute:
```shell
$ python3 train_mspeech.py
```
To start the test of this project, please execute:
```shell
$ python3 test_mspeech.py
```
Before testing, make sure the model file path filled in the code files exists.

ASRT API Server startup please execute:
```shell
$ python3 asrserver.py
```

Please note that after opening the API server, you need to use the client software corresponding to this ASRT project for voice recognition. For details, see the Wiki documentation [ASRT Client Demo](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki/ClientDemo).

If you want to train and use Model 251, make changes in the corresponding position of the `import SpeechModel` in the code files.

If there is any problem during the execution of the program or during use, it can be promptly put forward in the issue, and I will reply as soon as possible.



## Model

### Speech Model

CNN + LSTM/GRU + CTC

The maximum length of the input audio is 16 seconds, and the output is the corresponding Chinese pinyin sequence. 

* Questions about downloading trained models

The complete source program that includes trained model weights can be obtained from the archives of the various versions of the software released in the [releases](https://github.com/nl8590687/ASRT_SpeechRecognition/releases) page of Github.

### Language Model 

Maximum Entropy Hidden Markov Model Based on Probability Graph. 

The input is a Chinese pinyin sequence, and the output is the corresponding Chinese character text. 

## About Accuracy

At present, the best model can basically reach 80% of Pinyin correct rate on the test set. 

However, as the current international and domestic teams can achieve 98%, the accuracy rate still needs to be further improved. 

## Python libraries that need importing

* python_speech_features
* TensorFlow
* Keras
* Numpy
* wave
* matplotlib
* math
* Scipy
* h5py
* http
* urllib

## Data Sets 
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

Special thanks! Thanks to the predecessors' public voice data set. 

If the provided dataset link cannot be opened and downloaded, click this link [OpenSLR](http://www.openslr.org)

## Logs

Links: [Progress Logs](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/log.md)

## Contributors
[@zw76859420](https://github.com/zw76859420) 
@madeirak @ZJUGuoShuai @williamchenwl

@nl8590687 (repo owner)

[**Donate**](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki/donate)