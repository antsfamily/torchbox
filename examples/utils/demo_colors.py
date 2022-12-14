#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchbox as tb
import matplotlib.pyplot as plt

cmap = 'jet'
# cmap = 'hsv'
# cmap = 'hot'
# cmap = 'parula'
gray = tb.imread('data/images/oi/LenaGRAY256.png')
print(gray.shape)

rgb = tb.gray2rgb(gray, cmap, [0, 1], False)  # rgb --> double, [0, 1]
rgb = tb.gray2rgb(gray, cmap, [0, 255], False)  # rgb --> double, [0., 255.]
# rgb = tb.gray2rgb(gray, cmap, [0, 255], 'uint8')  # rgb --> uint8, [0, 255]

print(gray.shape, th.min(gray), th.max(gray), gray.dtype)
print(rgb.shape, th.min(rgb), th.max(rgb), rgb.dtype)

plt.figure()
plt.subplot(121)
plt.imshow(gray, cmap=tb.parula if cmap == 'parula' else cmap)
plt.subplot(122)
plt.imshow(rgb)
plt.show()
