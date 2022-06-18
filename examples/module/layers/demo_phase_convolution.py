#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-06 10:38:13
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torchbox as tb
import matplotlib.pyplot as plt


filename = '../../../data/images/Lotus512.png'
filename = '../../../data/images/LenaRGB512.tif'

x = tb.imread(filename)

x = x[..., 0:2].float()
x = x.unsqueeze(0).unsqueeze(0)

print(x.shape)

# pconv = tb.PhaseConv2d(1, 1, 511, 1, 255)
# pconv = tb.PhaseConv2d(1, 1, 7, 1, 3)
pconv = tb.PhaseConv2d(1, 1, 15, 1, 7)
# pconv = tb.ComplexPhaseConv2d(1, 1, 511, 1, 255)
# pconv = tb.ComplexPhaseConv2d(1, 1, 7, 1, 3)
# pconv = tb.ComplexPhaseConv2d(1, 1, 15, 1, 7)
pconv = tb.ComplexPhaseConv2d(1, 1, 3, 1, 1)

y = pconv(x)

x = x.squeeze(0).squeeze(0).detach().numpy()
y = y.squeeze(0).squeeze(0).detach().numpy()

print(np.sum(x - y))


plt.figure()
plt.subplot(231)
plt.imshow(x[..., 0])
plt.subplot(232)
plt.imshow(x[..., 1])
plt.subplot(233)
plt.imshow(np.arctan(x[..., 1]) / x[..., 0])

plt.subplot(234)
plt.imshow(y[..., 0])
plt.subplot(235)
plt.imshow(y[..., 1])
plt.subplot(236)
plt.imshow(np.arctan(y[..., 1]) / y[..., 0])
plt.show()


plt.figure()
plt.subplot(121)
plt.imshow(np.sqrt(np.sum(x**2, -1)))
plt.subplot(122)
plt.imshow(np.sqrt(np.sum(y**2, -1)))
plt.show()

