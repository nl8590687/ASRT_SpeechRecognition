# !/usr/bin/env python3
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
声学模型基础功能模板定义
"""
import os
import time
import random
import numpy as np

from utils.ops import get_edit_distance, read_wav_data
from utils.config import load_config_file, DEFAULT_CONFIG_FILENAME, load_pinyin_dict
from utils.thread import threadsafe_generator


class ModelSpeech:
    """
    语音模型类

    参数：
        speech_model: 声学模型类型 (BaseModel类) 实例对象
        speech_features: 声学特征类型(SpeechFeatureMeta类)实例对象
    """

    def __init__(self, speech_model, speech_features, max_label_length=64):
        self.data_loader = None
        self.speech_model = speech_model
        self.trained_model, self.base_model = speech_model.get_model()
        self.speech_features = speech_features
        self.max_label_length = max_label_length

    @threadsafe_generator
    def _data_generator(self, batch_size, data_loader):
        """
        数据生成器函数，用于Keras的generator_fit训练
        batch_size: 一次产生的数据量
        """
        labels = np.zeros((batch_size, 1), dtype=np.float)
        data_count = data_loader.get_data_count()
        index = 0

        while True:
            X = np.zeros((batch_size,) + self.speech_model.input_shape, dtype=np.float)
            y = np.zeros((batch_size, self.max_label_length), dtype=np.int16)
            input_length = []
            label_length = []

            for i in range(batch_size):
                wavdata, sample_rate, data_labels = data_loader.get_data(index)
                data_input = self.speech_features.run(wavdata, sample_rate)
                data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                # 必须加上模pool_size得到的值，否则会出现inf问题，然后提示No valid path found.
                # 但是直接加又可能会出现sequence_length <= xxx 的问题，因此不能让其超过时间序列长度的最大值，比如200
                pool_size = self.speech_model.input_shape[0] // self.speech_model.output_shape[0]
                inlen = min(data_input.shape[0] // pool_size + data_input.shape[0] % pool_size,
                            self.speech_model.output_shape[0])
                input_length.append(inlen)

                X[i, 0:len(data_input)] = data_input
                y[i, 0:len(data_labels)] = data_labels
                label_length.append([len(data_labels)])

                index = (index + 1) % data_count

            label_length = np.matrix(label_length)
            input_length = np.array([input_length]).T

            yield [X, y, input_length, label_length], labels

    def train_model(self, optimizer, data_loader, epochs=1, save_step=1, batch_size=16, last_epoch=0, call_back=None):
        """
        训练模型

        参数：
            optimizer：tensorflow.keras.optimizers 优化器实例对象
            data_loader：数据加载器类型 (SpeechData) 实例对象
            epochs: 迭代轮数
            save_step: 每多少epoch保存一次模型
            batch_size: mini batch大小
            last_epoch: 上一次epoch的编号，可用于断点处继续训练时，epoch编号不冲突
            call_back: keras call back函数
        """
        save_filename = os.path.join('save_models', self.speech_model.get_model_name(),
                                     self.speech_model.get_model_name())

        self.trained_model.compile(loss=self.speech_model.get_loss_function(), optimizer=optimizer)
        print('[ASRT] Compiles Model Successfully.')

        yielddatas = self._data_generator(batch_size, data_loader)

        data_count = data_loader.get_data_count()  # 获取数据的数量
        # 计算每一个epoch迭代的次数
        num_iterate = data_count // batch_size
        iter_start = last_epoch
        iter_end = last_epoch + epochs
        for epoch in range(iter_start, iter_end):  # 迭代轮数
            try:
                epoch += 1
                print('[ASRT Training] train epoch %d/%d .' % (epoch, iter_end))
                data_loader.shuffle()
                self.trained_model.fit_generator(yielddatas, num_iterate, callbacks=call_back)
            except StopIteration:
                print('[error] generator error. please check data format.')
                break

            if epoch % save_step == 0:
                if not os.path.exists('save_models'):  # 判断保存模型的目录是否存在
                    os.makedirs('save_models')  # 如果不存在，就新建一个，避免之后保存模型的时候炸掉
                if not os.path.exists(os.path.join('save_models', self.speech_model.get_model_name())):  # 判断保存模型的目录是否存在
                    os.makedirs(
                        os.path.join('save_models', self.speech_model.get_model_name()))  # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

                self.save_model(save_filename + '_epoch' + str(epoch))

        print('[ASRT Info] Model training complete. ')

    def load_model(self, filename):
        """
        加载模型参数
        """
        self.speech_model.load_weights(filename)

    def save_model(self, filename):
        """
        保存模型参数
        """
        self.speech_model.save_weights(filename)

    def evaluate_model(self, data_loader, data_count=-1, out_report=False, show_ratio=True, show_per_step=100):
        """
        评估检验模型的识别效果
        """
        data_nums = data_loader.get_data_count()

        if data_count <= 0 or data_count > data_nums:  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = data_nums

        try:
            ran_num = random.randint(0, data_nums - 1)  # 获取一个随机数
            words_num = 0
            word_error_num = 0

            nowtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            if out_report:
                txt_obj = open('Test_Report_' + data_loader.dataset_type + '_' + nowtime + '.txt', 'w',
                               encoding='UTF-8')  # 打开文件并读入
                txt_obj.truncate((data_count + 1) * 300)  # 预先分配一定数量的磁盘空间，避免后期在硬盘中文件存储位置频繁移动，以防写入速度越来越慢
                txt_obj.seek(0)  # 从文件首开始

            txt = ''
            i = 0
            while i < data_count:
                wavdata, fs, data_labels = data_loader.get_data((ran_num + i) % data_nums)  # 从随机数开始连续向后取一定数量数据
                data_input = self.speech_features.run(wavdata, fs)
                data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                # 数据格式出错处理 开始
                # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
                if data_input.shape[0] > self.speech_model.input_shape[0]:
                    print('*[Error]', 'wave data lenghth of num', (ran_num + i) % data_nums, 'is too long.',
                          'this data\'s length is', data_input.shape[0],
                          'expect <=', self.speech_model.input_shape[0],
                          '\n A Exception raise when test Speech Model.')
                    i += 1
                    continue
                # 数据格式出错处理 结束

                pre = self.predict(data_input)

                words_n = data_labels.shape[0]  # 获取每个句子的字数
                words_num += words_n  # 把句子的总字数加上
                edit_distance = get_edit_distance(data_labels, pre)  # 获取编辑距离
                if edit_distance <= words_n:  # 当编辑距离小于等于句子字数时
                    word_error_num += edit_distance  # 使用编辑距离作为错误字数
                else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
                    word_error_num += words_n  # 就直接加句子本来的总字数就好了

                if i % show_per_step == 0 and show_ratio:
                    print('[ASRT Info] Testing: ', i, '/', data_count)

                txt = ''
                if out_report:
                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(pre) + '\n'
                    txt += '\n'
                    txt_obj.write(txt)

                i += 1

            # print('*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率：', word_error_num / words_num * 100, '%')
            print('*[ASRT Test Result] Speech Recognition ' + data_loader.dataset_type + ' set word error ratio: ',
                  word_error_num / words_num * 100, '%')
            if out_report:
                txt = '*[ASRT Test Result] Speech Recognition ' + data_loader.dataset_type + ' set word error ratio: ' + str(
                    word_error_num / words_num * 100) + ' %'
                txt_obj.write(txt)
                txt_obj.truncate()  # 去除文件末尾剩余未使用的空白存储字节
                txt_obj.close()

        except StopIteration:
            print('[ASRT Error] Model testing raise a error. Please check data format.')

    def predict(self, data_input):
        """
        预测结果

        返回语音识别后的forward结果
        """
        return self.speech_model.forward(data_input)

    def recognize_speech(self, wavsignal, fs):
        """
        最终做语音识别用的函数，识别一个wav序列的语音
        """
        # 获取输入特征
        data_input = self.speech_features.run(wavsignal, fs)
        data_input = np.array(data_input, dtype=np.float)
        # print(data_input,data_input.shape)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        r1 = self.predict(data_input)
        # 获取拼音列表
        list_symbol_dic, _ = load_pinyin_dict(load_config_file(DEFAULT_CONFIG_FILENAME)['dict_filename'])

        r_str = []
        for i in r1:
            r_str.append(list_symbol_dic[i])

        return r_str

    def recognize_speech_from_file(self, filename):
        """
        最终做语音识别用的函数，识别指定文件名的语音
        """
        wavsignal, sample_rate, _, _ = read_wav_data(filename)
        r = self.recognize_speech(wavsignal, sample_rate)
        return r

    @property
    def model(self):
        """
        返回tf.keras model
        """
        return self.trained_model
