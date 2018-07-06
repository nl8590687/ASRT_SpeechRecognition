# A Deep-Learning-Based Chinese Speech Recognition System

ReadMe Language [中文版](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/README.md) [English](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/README_EN.md) 

## Introduction

This project uses Keras, TensorFlow based on deep convolutional neural network and long-short memory neural network, attention mechanism and CTC to implement.

[View this project's wiki page](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki) (In progress..)

The project can now be properly trained.

After cloning a repository through git, you need to copy all the files in the datalist directory to the dataset directory, that is, put them together with the data set.

```shell
$ cp -rf datalist/* dataset/
```

Currently available models are 22, 24 and 25

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

If you want to train and use Model 251, make changes in the corresponding position of the `import SpeechModel` in the code files.

If there is any problem during the execution of the program or during use, it can be promptly put forward in the issue, and I will reply as soon as possible.

You can check the [FAQ](https://github.com/nl8590687/ASRT_SpeechRecognition/wiki/issues) first before asking questions.

## Model

### Speech Model

CNN + LSTM/GRU + CTC

* Questions about downloading trained models

The complete source program can be obtained from the archives of the various versions of the software released in the release page of Github.

### Language Model 

Maximum Entropy Hidden Markov Model Based on Probability Graph. 

## About Accuracy

At present, the best model can basically reach 80% of Pinyin correct rate on the test set. 

However, as the current international and domestic teams can achieve 97%, the accuracy rate still needs to be further improved. 

* At present, one solution that can continue to improve the accuracy rate is correcting data set labeling errors, especially the ST-CMDS error in the syllable file. There is a certain percentage of errors in the label. If you have see this and you have the will to help correct some of the data tagging mistakes by own ability, I will be very welcome. It can be corrected by submitting a Pull Request, and you will be on the list of contributors of this repo.

Samples: `不是： bu4 shi4 -> bu2 shi4` `一个：yi1 ge4 -> yi2 ge4` `了解：le5 jie3 -> liao3 jie3`

* Corrected part:

ST-CMDS

train:  20170001P00001A    20170001P00001I    20170001P00002A

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

## Data Sets 
* Tsinghua University THCHS30 Chinese voice data set

data_thchs30.tgz 
<http://www.openslr.org/resources/18/data_thchs30.tgz>

test-noise.tgz 
<http://www.openslr.org/resources/18/test-noise.tgz>

resource.tgz 
<http://www.openslr.org/resources/18/resource.tgz>

* Free ST Chinese Mandarin Corpus

ST-CMDS-20170001_1-OS.tar.gz 
<http://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz>

Special thanks! Thanks to the predecessors' public voice data set. 

If the provided dataset link cannot be opened and downloaded, click this link [OpenSLR] (http://www.openslr.org)

## Logs

Links: [Progress Logs](https://github.com/nl8590687/ASRT_SpeechRecognition/blob/master/log.md)

## Contributors
@ZJUGuoShuai @williamchenwl

@nl8590687 (repo owner)