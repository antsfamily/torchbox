#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#

import torchbox as tb
import matplotlib.pyplot as plt

datafolder = tb.data_path('optical')
x = tb.imread(datafolder + 'LenaRGB512.tif')

# ratio1 = [1, 1]
# ratio1 = [0.8, 1]
# ratio1 = [0.5, 1]
ratio1 = [0.25, 1]
# ratio1 = [0.125, 1]
# ratio2 = [1, 0.8]
# ratio2 = [1, 0.5]
# ratio2 = [1, 0.25]
# ratio2 = [1, 0.125]
ratio2 = [0.5, 0.5]
smode = 'uniform'
# smode = 'random'
# smode = 'random2'  # has problem
omode = 'zero'

y1, mask1 = tb.dnsampling(x, ratio=ratio1, dim=[0, 1], smode=smode, omode=omode, retall=True)
y2, mask2 = tb.dnsampling(x, ratio=ratio2, dim=[0, 1], smode=smode, omode=omode, retall=True)

H, W = mask1.shape[0], mask1.shape[1]
print(mask1.sum() / (H * W * 1.), ratio1[0] * ratio1[1])
print(mask2.sum() / (H * W * 1.), ratio2[0] * ratio2[1])

plt.figure()
plt.subplot(221)
plt.imshow(mask1)
plt.subplot(222)
plt.imshow(mask2)
plt.subplot(223)
plt.imshow(y1)
plt.subplot(224)
plt.imshow(y2)
plt.show()

