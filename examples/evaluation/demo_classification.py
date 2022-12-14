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

T = th.tensor([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 5])
P = th.tensor([1, 2, 3, 4, 1, 6, 3, 2, 1, 4, 5, 6, 1, 2, 1, 4, 5, 6, 1, 5])

print(tb.accuracy(P, T))
# print(tb.categorical2onehot(T))

C = tb.confusion(P, T, cmpmode='...')
print(C)
C = tb.confusion(P, T, cmpmode='@')
print(C)
print(tb.kappa(C))
print(tb.kappa(C.T))

plt = tb.plot_confusion(C, cmap=None)
plt = tb.plot_confusion(C, cmap='summer')
plt.show()

