#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#

import math
import torch as th
import torchbox as tb
import matplotlib.pyplot as plt


datafolder = tb.data_path('optical')
x = tb.imread(datafolder + 'LenaRGB512.tif')

x = x[0:9, 0:9, :]

xshape = x.shape
xshape = xshape[:2]

n, size = None, (3, 3)

y = tb.tensor2patch(x, n=n, size=size, dim=(0, 1), step=(1, 1), shake=(0, 0), mode='slidegrid', seed=2020)

print(y.shape)
