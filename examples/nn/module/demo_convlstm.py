#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : demo_convlstm.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Feb 19 2023
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
import torchbox as tb


tb.setseed(seed=2023, target='torch')

T, B, C, H, W = 10, 6, 2, 18, 18
x = th.randn(T, B, C, H, W)

# ===way1
tb.setseed(seed=2023, target='torch')
lstm = tb.ConvLSTM(rank=2, in_channels=[C, 4], out_channels=[4, 4], kernel_size=[3, 3], stride=[1, 1], padding=['same', 'same'])
print(lstm.cells[0].in_convc.weight.sum(), lstm.cells[0].rnn_convc.weight.sum())
print(lstm.cells[1].in_convc.weight.sum(), lstm.cells[1].rnn_convc.weight.sum())

# ===way2
tb.setseed(seed=2023, target='torch')
cell1 = tb.ConvLSTMCell(rank=2, in_channels=C, out_channels=4, kernel_size=3, stride=1, padding='same')
cell2 = tb.ConvLSTMCell(rank=2, in_channels=4, out_channels=4, kernel_size=3, stride=1, padding='same')

print(cell1.in_convc.weight.sum(), cell1.rnn_convc.weight.sum())
print(cell2.in_convc.weight.sum(), cell2.rnn_convc.weight.sum())

# ===way1
y, (h, c) = lstm(x, None)
h = th.stack(h, dim=0)
c = th.stack(c, dim=0)

print(y.shape, y.sum(), h.shape, h.sum(), c.shape, c.sum())

# ===way2
h1, c1 = None, None
h2, c2 = None, None
y = []
for t in range(x.shape[0]):
    h1, c1 = cell1(x[t, ...], (h1, c1))
    h2, c2 = cell2(h1, (h2, c2))
    y.append(h2)
y = th.stack(y, dim=0)
h = th.stack((h1, h2), dim=0)
c = th.stack((c1, c2), dim=0)
print(y.shape, y.sum(), h.shape, h.sum(), c.shape, c.sum())
