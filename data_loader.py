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
加载语音数据集用的数据加载器的定义
"""
import os
import random
import numpy as np
from utils.config import load_config_file, DEFAULT_CONFIG_FILENAME, load_pinyin_dict
from utils.ops import read_wav_data


class DataLoader:
    """
    数据加载器

    参数：\\
        config: 配置信息字典
        dataset_type: 要加载的数据集类型，包含('train', 'dev', 'test')三种
    """
    def __init__(self, dataset_type:str):
        self.dataset_type = dataset_type

        self.data_list = list()
        self.wav_dict = dict()
        self.label_dict = dict()
        self.pinyin_list = list()
        self.pinyin_dict = dict()
        self._load_data()

    def _load_data(self):
        config = load_config_file(DEFAULT_CONFIG_FILENAME)

        self.pinyin_list, self.pinyin_dict = load_pinyin_dict(config['dict_filename'])

        for index in range(len(config['dataset'][self.dataset_type])):
            filename_datalist = config['dataset'][self.dataset_type][index]['data_list']
            filename_datapath = config['dataset'][self.dataset_type][index]['data_path']
            with open(filename_datalist, 'r', encoding='utf-8') as file_pointer:
                lines = file_pointer.read().split('\n')
                for line in lines:
                    if len(line) == 0:
                        continue
                    tokens = line.split(' ')
                    self.data_list.append(tokens[0])
                    self.wav_dict[tokens[0]] = os.path.join(filename_datapath, tokens[1])

            filename_labellist = config['dataset'][self.dataset_type][index]['label_list']
            with open(filename_labellist, 'r', encoding='utf-8') as file_pointer:
                lines = file_pointer.read().split('\n')
                for line in lines:
                    if len(line) == 0:
                        continue
                    tokens = line.split(' ')
                    self.label_dict[tokens[0]] = tokens[1:]

    def get_data_count(self) -> int:
        """
        获取数据集总数量
        """
        return len(self.data_list)

    def get_data(self, index:int) -> tuple:
        """
        按下标获取一条数据
        """
        mark = self.data_list[index]

        wav_signal, sample_rate, _, _ = read_wav_data(self.wav_dict[mark])
        labels = list()
        for item in self.label_dict[mark]:
            if len(item) == 0:
                continue
            labels.append(self.pinyin_dict[item])

        data_label = np.array(labels)
        return wav_signal, sample_rate, data_label

    def shuffle(self) -> None:
        """
        随机打乱数据
        """
        random.shuffle(self.data_list)
