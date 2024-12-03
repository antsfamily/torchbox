#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : demo_classification.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Wed Dec 14 2022
# @version   : 0.0
# @license   : The GNU General Public License (GPL) v3.0
# @note      : 
# 
# The GNU General Public License (GPL) v3.0
# Copyright (C) 2013- Zhi Liu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import torch as th
import torchbox as tb


seed = 2023
dat_file = '/mnt/e/DataSets/wifi/csi/wifi_channel_pc1.mat'

H = tb.loadmat(dat_file)['H']  # TxSCxBSxMS
H = th.from_numpy(H).to(th.complex64)
Nt = H.shape[0]

n = 1
# using the channel of last time as the predicted
H, Hp1 = H[n:], H[:Nt-n]

# using the noised version of the cahnnel as the predicted
Hp2 = tb.awgns(H, snrv=5, cdim=None, dim=(-3, -2, -1), seed=seed)  # 5dB
print(H.shape, Hp1.shape, Hp2.shape)

metric = tb.ChnlCapCor(EsN0=30, rank=2, cdim=None, dim=(-3, -2, -1), reduction='mean')
metric.updategt(H)

print(metric.forward(H))
print(metric.forward(Hp1))
print(metric.forward(Hp2))

