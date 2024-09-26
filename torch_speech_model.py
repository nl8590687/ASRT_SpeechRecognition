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
import time

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from data_loader import DataLoader
from speech_features.speech_features import SpeechFeatureMeta


class SpeechDataset(Dataset):
    def __init__(self, data_loader, speech_features, input_shape, max_label_length):
        self.data_loader = data_loader
        self.input_shape = input_shape
        self.speech_features = speech_features
        self.max_label_length = max_label_length
        self.data_count = self.data_loader.get_data_count()

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

        # 初始化标签数组，填充到 `max_label_length` 大小
        y = torch.zeros(self.max_label_length, dtype=torch.int16)
        y[:len(data_labels)] = torch.tensor(data_labels, dtype=torch.int16) + 1

        # 转换为 PyTorch 张量
        input_length = torch.tensor((inlen,), dtype=torch.float32)
        label_length = torch.tensor((len(data_labels),), dtype=torch.float32)
        return x, y, input_length, label_length


class ModelSpeech:
    def __init__(self, speech_model: torch.nn.Module, speech_features: SpeechFeatureMeta, max_label_length: int = 64):
        """模型初始化"""
        self.speech_model = speech_model
        self.trained_model = speech_model.get_model()
        self.speech_features = speech_features
        self.max_label_length = max_label_length

    def train(self, data_loader: DataLoader, epochs: int, batch_size: int, optimizer: torch.optim.Optimizer,
              device: str = 'cpu'):
        """训练模型"""
        speechdata = SpeechDataset(data_loader, self.speech_features, input_shape=self.speech_model.input_shape,
                                   max_label_length=self.max_label_length)
        self.trained_model.to(device)
        print('[ASRT] torch model successfully initialized to device: {}'.format(device))
        data_loader = TorchDataLoader(speechdata, batch_size=batch_size, shuffle=True)
        model = self.speech_model
        for epoch in range(epochs):
            print('[ASRT] Epoch {}/{}'.format(epoch + 1, epochs))
            epoch_loss = 0.0
            iter_index = 0
            t0 = time.time()
            for batch in data_loader:
                x, y, input_length, label_length = batch
                x = x.to(device)
                y = y.to(device)
                input_length = input_length.to(device).long()
                label_length = label_length.to(device).long()

                optimizer.zero_grad()
                y_pred = model(x)
                loss = model.compute_loss(y_pred, y, input_length, label_length)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                iter_index += 1
                t1 = time.time()
                predict_total_time = (t1-t0)*len(data_loader)/iter_index
                predict_remain_time = predict_total_time - (t1-t0)
                cur_batch_loss = loss.item()
                cur_avg_loss = epoch_loss / iter_index
                print("[ASRT]", f"{predict_remain_time:.2f}/{predict_total_time:.2f} s,",
                      f"step {iter_index}/{len(data_loader)},", f"current loss: {cur_batch_loss:.4f}",
                      f"avg loss: {cur_avg_loss:.4f}", end="\r")

            save_filename = os.path.join('save_models_torch', f"{self.speech_model.get_model_name()}_epoch{epoch+1}.pth")
            self.save_weight(save_filename)
            avg_loss = epoch_loss / len(data_loader)
            total_time = time.time()-t0
            avg_time_per_step = total_time / len(data_loader)
            print("[ASRT]", f"epoch {epoch + 1}/{epochs},", f"time cost: {total_time:.2f} s,",
                  f"{avg_time_per_step:.2f} s/step", f"avg loss: {avg_loss:.4f}")

    def save_weight(self, filename: str):
        save_filename = os.path.join('save_models_torch', filename + ".pth")
        torch.save(self.speech_model.state_dict(), save_filename)

    def load_weight(self, filepath: str):
        self.speech_model.load_state_dict(torch.load(filepath))
