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


mode = 'abs'
# mode = None
print('---compare eigen vector correlation (complex in real)')
G = th.randn(2, 3, 2, 64, 4)
P = th.randn(2, 3, 2, 64, 4)
print(tb.eigveccor(P, G, npcs=4, mode=mode, cdim=2, sdim=-1, fdim=-2, keepdim=False, reduction=None))
print(tb.eigveccor(P, G, npcs=4, mode=mode, cdim=2, sdim=-1, fdim=-2, keepdim=False, reduction='sum'))
print(tb.eigveccor(P, G, npcs=4, mode=mode, cdim=2, sdim=-1, fdim=-2, keepdim=False, reduction='mean'))

print('---compare eigen vector correlation (complex in complex)')
G = tb.r2c(G, cdim=2, keepdim=False)
P = tb.r2c(P, cdim=2, keepdim=False)
print(tb.eigveccor(P, G, npcs=4, mode=mode, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction=None))
print(tb.eigveccor(P, G, npcs=4, mode=mode, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction='sum'))
print(tb.eigveccor(P, G, npcs=4, mode=mode, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction='mean'))


dat_file = '/mnt/e/DataSets/wifi/csi/wifi_channel_pc1.mat'

H = tb.loadmat(dat_file)['H']  # TxSCxBSxMS
H = th.from_numpy(H).to(th.complex64)
Nt = H.shape[0]
n = 1
Hp, H = H[n:], H[:Nt-n]
print(H.shape)
print(tb.eigveccor(Hp, H, npcs=2, mode=mode, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction='sum'))
print(tb.eigveccor(Hp, H, npcs=2, mode=mode, cdim=None, sdim=-1, fdim=-2, keepdim=False, reduction='mean'))
