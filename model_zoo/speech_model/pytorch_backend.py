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
若干 pytorch版声学模型模型的定义和实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as function


class SpeechModel251BN(nn.Module):
    def __init__(self, input_shape: tuple = (1600, 200, 1), output_size: int = 1428):
        super(SpeechModel251BN, self).__init__()

        self.input_shape = input_shape
        self._pool_size = 8
        self._model_name = 'SpeechModel251bn'
        self.output_shape = (input_shape[0] // self._pool_size, output_size)

        # block 1
        self.conv0 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same', dtype=torch.bfloat16)
        self.conv0.weight = torch.nn.init.kaiming_normal_(self.conv0.weight)
        self.bn0 = nn.BatchNorm2d(32, eps=0.0002)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same', dtype=torch.bfloat16)
        self.conv1.weight = torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(32, eps=0.0002)

        # block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same', dtype=torch.bfloat16)
        self.conv2.weight = torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(64, eps=0.0002)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same', dtype=torch.bfloat16)
        self.conv3.weight = torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(64, eps=0.0002)

        # block 3
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same', dtype=torch.bfloat16)
        self.conv4.weight = torch.nn.init.kaiming_normal_(self.conv4.weight)
        self.bn4 = nn.BatchNorm2d(128, eps=0.0002)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same', dtype=torch.bfloat16)
        self.conv5.weight = torch.nn.init.kaiming_normal_(self.conv5.weight)
        self.bn5 = nn.BatchNorm2d(128, eps=0.0002)

        # block 4
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same', dtype=torch.bfloat16)
        self.conv6.weight = torch.nn.init.kaiming_normal_(self.conv6.weight)
        self.bn6 = nn.BatchNorm2d(128, eps=0.0002)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same', dtype=torch.bfloat16)
        self.conv7.weight = torch.nn.init.kaiming_normal_(self.conv7.weight)
        self.bn7 = nn.BatchNorm2d(128, eps=0.0002)

        # block 5
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same', dtype=torch.bfloat16)
        self.conv8.weight = torch.nn.init.kaiming_normal_(self.conv8.weight)
        self.bn8 = nn.BatchNorm2d(128, eps=0.0002)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same', dtype=torch.bfloat16)
        self.conv9.weight = torch.nn.init.kaiming_normal_(self.conv9.weight)
        self.bn9 = nn.BatchNorm2d(128, eps=0.0002)

        self.dense0 = nn.Linear(input_shape[1]//8*128, 128, dtype=torch.bfloat16)
        self.dense0.weight = torch.nn.init.kaiming_normal_(self.dense0.weight)

        self.dense1 = nn.Linear(128, output_size, dtype=torch.bfloat16)
        self.dense1.weight = torch.nn.init.kaiming_normal_(self.dense1.weight)

        self.ctc_loss = nn.CTCLoss(blank=0)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)

        # block 1
        x = function.relu(self.bn0(self.conv0(x)))
        x = function.relu(self.bn1(self.conv1(x)))
        x = function.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        # block 2
        x = function.relu(self.bn2(self.conv2(x)))
        x = function.relu(self.bn3(self.conv3(x)))
        x = function.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        # block 3
        x = function.relu(self.bn4(self.conv4(x)))
        x = function.relu(self.bn5(self.conv5(x)))
        x = function.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        # block 4
        x = function.relu(self.bn6(self.conv6(x)))
        x = function.relu(self.bn7(self.conv7(x)))
        x = function.max_pool2d(x, kernel_size=(1, 1), stride=(1, 1))

        # block 5
        x = function.relu(self.bn8(self.conv8(x)))
        x = function.relu(self.bn9(self.conv9(x)))
        x = function.max_pool2d(x, kernel_size=(1, 1), stride=(1, 1))

        x = x.reshape(x.size(0), -1, x.size(3))
        x = x.permute(0, 2, 1)
        x = function.relu(self.dense0(x))
        x = function.log_softmax(self.dense1(x), dim=2)
        return x  # (batch, time, classes)

    def compute_loss(self, y_pred, labels, input_length, label_length):
        y_pred = y_pred.permute(1, 0, 2)  # (time, batch, classes)
        y_pred = y_pred.float()
        loss = self.ctc_loss(y_pred, labels, input_length, label_length)
        return loss

    def get_model(self):
        return self

    def get_model_name(self) -> str:
        return self._model_name
