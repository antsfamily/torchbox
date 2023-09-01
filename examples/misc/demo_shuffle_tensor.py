#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#

import torch as th
import torchbox as tb

tb.setseed(2020, 'torch')
x = th.randint(1000, (20, 3, 4))
y1, idx1 = tb.shuffle_tensor(x, dim=0, groups=4, mode='intra', retall=True)
y2, idx2 = tb.shuffle_tensor(x, dim=0, groups=4, mode='inter', retall=True)
y3, idx3 = tb.shuffle_tensor(x, dim=0, groups=4, mode='whole', retall=True)

print(x.shape)
print(y1.shape)
print(y2.shape)
print(y3.shape)
print(idx1)
print(idx2)
print(idx3)

print((y1 - x[idx1]).sum())
print((y2 - x[idx2]).sum())
print((y3 - x[idx3]).sum())
