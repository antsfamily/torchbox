#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : edge.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Nov 27 2019
# @version   : 0.0
# @license   : The Apache License 2.0
# @note      : 
# 
# The Apache 2.0 License
# Copyright (C) 2013- Zhi Liu
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#

import torch
from torch.autograd import Variable
import torch.nn.functional as F


class EdgeDetector(torch.nn.Module):
    def __init__(self):
        super(EdgeDetector, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        blur = torch.FloatTensor(
            [[1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0]]
        )
        blur *= float(1.0 / 9.0)

        weightY = torch.FloatTensor(
            [[1.0, 0.0, -1.0],
             [2.0, 0.0, -2.0],
             [1.0, 0.0, -1.0]]
        )
        weightX = weightY.t()
        self.blur = Variable(blur.unsqueeze_(0).unsqueeze_(0)).to(self.device)
        self.kernelY = Variable(weightY.unsqueeze_(0).unsqueeze_(0)).to(self.device)
        self.kernelX = Variable(weightX.unsqueeze_(0).unsqueeze_(0)).to(self.device)

    def forward(self, image):
        data = image.clone()
        input = Variable(data).to(self.device)
        blurred = F.conv2d(input, self.blur, stride=1, padding=1)
        Y = F.conv2d(blurred, self.kernelY, stride=1, padding=1)
        X = F.conv2d(blurred, self.kernelX, stride=1, padding=1)
        # out = torch.sqrt(X * X + Y * Y)
        out = torch.abs(X) + torch.abs(Y)
        return out


class EdgeFeatureExtractor(torch.nn.Module):
    def __init__(self, Ci):
        super(EdgeFeatureExtractor, self).__init__()
        self.filter1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=Ci, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        self.filter2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        self.filter3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h = self.filter1(x)
        feature_1 = h
        h = self.filter2(h)
        feature_2 = h
        h = self.filter3(h)
        feature_3 = h
        return feature_1, feature_2, feature_3
