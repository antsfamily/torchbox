#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : consistency_layers.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Nov 27 2019
# @version   : 0.0
# @license   : The GNU General Public License (GPL) v3.0
# @note      : 
# 
# The GNU General Public License (GPL) v3.0
# Copyright (C) 2013- Zhi Liu
#
# This file is part of torchbox.
#
# torchbox is free software: you can redistribute it and/or modify it under the 
# terms of the GNU General Public License as published by the Free Software Foundation, 
# either version 3 of the License, or (at your option) any later version.
#
# torchbox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with torchbox. 
# If not, see <https://www.gnu.org/licenses/>. 
#

import torch as th


class DataConsistency2d(th.nn.Module):

    def __init__(self, ftaxis=(-2, -1), mixrate=1.0, isfft=True):
        super(DataConsistency2d, self).__init__()
        self.ftaxis = ftaxis
        self.mixrate = mixrate
        self.isfft = isfft

    def forward(self, x, y, mask):
        d = x.dim()
        maskshape = [1] * d
        for a in self.ftaxis:
            maskshape[a] = x.shape[a]
        mask = mask * self.mixrate
        mask = mask.reshape(maskshape)
        if self.isfft:
            xf = th.fft.fft2(x, s=None, dim=self.ftaxis, norm=None)
            yf = th.fft.fft2(y, s=None, dim=self.ftaxis, norm=None)

        xf = xf * mask
        yf = yf * (1.0 - mask)

        return th.fft.ifft2(xf + yf, s=None, dim=self.ftaxis, norm=None)


if __name__ == '__main__':

    N, C, H, W = 5, 2, 3, 4
    x = th.randn(N, C, H, W)
    y = th.randn(N, C, H, W)
    mask = th.rand(H, W)
    mask[mask < 0.5] = 0
    mask[mask > 0.5] = 1
    # mask = th.ones(H, W)

    dc = DataConsistency2d(ftaxis=(-2, -1), mixrate=1., isfft=True)

    z = dc(x, y, mask)

    print(x, 'x')
    print(y, 'y')
    print(z.abs(), 'z')
