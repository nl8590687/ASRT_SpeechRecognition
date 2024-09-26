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
@author: nl8590687
pytorch版声学模型训练脚本入口
"""
from torch import optim

from torch_speech_model import *
from speech_features import SpecAugment
from data_loader import DataLoader
from model_zoo.speech_model.pytorch_backend import SpeechModel251BN

if __name__ == "__main__":
    feat = SpecAugment()
    data_loader = DataLoader('train')

    model = SpeechModel251BN()
    speechModel = ModelSpeech(model, feat, max_label_length=64)
    print(model)

    # speechModel.load_weight(os.path.join('save_models_torch', model.get_model_name()+"_save.pth"))
    speechModel.train(data_loader, epochs=10, batch_size=16, optimizer=optim.Adam(model.parameters(), lr=0.001),
                      device="cuda:0")
    speechModel.save_weight(model.get_model_name()+"_save")
