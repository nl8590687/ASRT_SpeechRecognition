#!/usr/bin/env python3
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
@author: nl8590687 / Evelynn-n
声学模型基础功能模板定义
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader


class SpeechDataset(Dataset):
    def __init__(self, data_loader, speech_features, input_shape, max_label_length, device='cpu'):
        self.data_loader = data_loader
        self.input_shape = input_shape
        self.speech_features = speech_features
        self.max_label_length = max_label_length
        self.data_count = self.data_loader.get_data_count()
        self.device = device

    def __len__(self):
        return self.data_count
    
    def __getitem__(self, index):
        wav_data, sample_rate, data_labels = self.data_loader.get_data(index)

        # 提取特征
        data_input = self.speech_features.run(wav_data, sample_rate)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)

        # 计算输入长度，确保不超出最大序列长度
        pool_size = self.input_shape[0] // (self.input_shape[0] // 8)
        inlen = min(data_input.shape[0] // pool_size + data_input.shape[0] % pool_size, self.input_shape[0] // 8)

        # 初始化输入特征数组，填充到 `input_shape` 大小
        x = torch.zeros(self.input_shape)
        x[:len(data_input)] = torch.tensor(data_input, dtype=torch.float32)
        x = x.permute(2, 0, 1)

        # 初始化标签数组，填充到 `max_label_length` 大小
        y = torch.zeros(self.max_label_length, dtype=torch.int16)
        y[:len(data_labels)] = torch.tensor(data_labels, dtype=torch.int16)

        # 转换为 PyTorch 张量
        input_length = torch.tensor([inlen], dtype=torch.float32)
        label_length = torch.tensor([len(data_labels)], dtype=torch.float32)
        return x, y, input_length, label_length


class ModelSpeech:
    def __init__(self, speech_model, speech_features, max_label_length=64):
        """模型初始化"""
        self.speech_model = speech_model
        self.trained_model = speech_model.get_model()
        self.speech_features = speech_features
        self.max_label_length = max_label_length

    def train(self, data_loader, epochs, batch_size, optimizer, save_step=1, last_epoch=0, device='cpu'):
        """训练模型"""
        save_filename = os.path.join('save_models_torch', self.speech_model.get_model_name() + '.pth')
        self.trained_model.to(device)
        print('[ASRT] torch model successfully initialized to device: {}'.format(device))
        data_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=True)
        model = self.speech_model
        for epoch in range(epochs):
            print('[ASRT] Epoch {}/{}'.format(epoch+1, epochs))
            epoch_loss = 0.0
            for batch in data_loader:
                x, y, input_length, label_length = batch
                x = x.to(device)
                y = y.to(device)
                input_length = input_length.to(device).unsqueeze(1).long()
                label_length = label_length.to(device).unsqueeze(1).long()
                optimizer.zero_grad()
                y_pred = model(x)
                # print(y_pred.shape, y.shape, input_length.shape, label_length.shape)
                loss = model.compute_loss(y_pred, y, input_length, label_length)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                avg_loss = epoch_loss / len(data_loader)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
