#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
# @Note    :

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
