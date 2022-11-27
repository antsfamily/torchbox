#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : pool.py
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

import torch as th


class MeanSquarePool2d(th.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(MeanSquarePool2d, self).__init__()
        self.kernel_size = kernel_size
        if type(kernel_size) is int or len(kernel_size) == 1:
            self.scale = kernel_size**2
        else:
            self.scale = th.prod(kernel_size)

        self.pool = th.nn.AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def forward(self, x):

        return th.sqrt(self.pool(x.pow(2)) / self.scale)


class PnormPool2d(th.nn.Module):
    def __init__(self, kernel_size, p=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super(PnormPool2d, self).__init__()
        self.p = p
        if type(kernel_size) is int or len(kernel_size) == 1:
            self.scale = kernel_size**2
        else:
            self.scale = th.prod(kernel_size)
        self.pool = th.nn.AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def forward(self, x):

        return (self.pool(th.abs(x).pow(self.p)) / self.scale).pow(1. / self.p)


if __name__ == '__main__':

    X = th.randn(2, 3, 8, 8)
    print(X)
    print(X**2)
    print(X.pow(2))

    pool1 = MeanSquarePool2d(3, stride=1, padding=0)
    pool2 = PnormPool2d(3, p=2, stride=1, padding=0)

    Xp1 = pool1(X)
    Xp2 = pool2(X)

    print(Xp1.size(), Xp2.size())

    print(Xp1 == Xp2)
