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

xshape = x.shape
xshape = xshape[:2]

n, size = 64, (64, 64)

y1 = tb.tensor2patch(x, n=n, size=size, dim=(0, 1), step=(1, 1), shake=(0, 0), mode='randperm', seed=2020)
y2 = tb.tensor2patch(x, n=n, size=size, dim=(0, 1), step=(64, 64), shake=(0, 0), mode='randgrid', seed=2020)

print(y1.shape, y2.shape)

Y1 = tb.patch2tensor(y1, size=xshape, dim=(1, 2), mode='nfirst')
Y2 = tb.patch2tensor(y2, size=xshape, dim=(1, 2), mode='nfirst')

plt.figure()
plt.subplot(121)
plt.imshow(Y1)
plt.title('randperm')
plt.subplot(122)
plt.imshow(Y2)
plt.title('randgrid')


osize = list(y1.shape[1:])
M = int(math.sqrt(n))
osize[0] = int(M * size[0])
osize[1] = int(M * size[1])

Y1 = th.zeros(osize, dtype=x.dtype)
Y2 = th.zeros(osize, dtype=x.dtype)

print(y1.shape, Y1.shape)
print(y2.shape, Y2.shape)

for i in range(M):
    for j in range(M):
        Y1[i * size[0]:(i + 1) * size[0], j * size[1]:(j + 1) * size[1], ...] = y1[i * M + j, ...]
        Y2[i * size[0]:(i + 1) * size[0], j * size[1]:(j + 1) * size[1], ...] = y2[i * M + j, ...]

plt.figure()
plt.subplot(121)
plt.imshow(Y1)
plt.title('randperm')
plt.subplot(122)
plt.imshow(Y2)
plt.title('randgrid')
plt.show()

n, size = 64, (64, 64)
shake1, shake2 = (0, 0), (64, 64)

y1 = tb.tensor2patch(x, n=n, size=size, dim=(0, 1), step=(64, 64), shake=shake1, mode='slidegrid', seed=2020)
y2 = tb.tensor2patch(x, n=n, size=size, dim=(0, 1), step=(64, 64), shake=shake2, mode='slidegrid', seed=2020)

print(y1.shape, y2.shape)

Y1 = tb.patch2tensor(y1, size=xshape, dim=(1, 2), mode='nfirst')
Y2 = tb.patch2tensor(y2, size=xshape, dim=(1, 2), mode='nfirst')

plt.figure()
plt.subplot(121)
plt.imshow(Y1)
plt.title('slidegrid ' + str(shake1))
plt.subplot(122)
plt.imshow(Y2)
plt.title('slidegrid ' + str(shake2))
plt.show()
