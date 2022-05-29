#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#

import torchbox as tb
import matplotlib.pyplot as plt

filename = '../../data/images/Lotus512.png'
filename = '../../data/images/LenaRGB512.tif'
filename = '../../data/images/LenaGRAY512.tif'

x0 = tb.imread(filename)

x = x0 + 1j * x0

y = tb.ct2rt(x, axis=0)
z = tb.rt2ct(y, axis=0)

x, y, z = x.real, y.real, z.real

print(x.shape, y.shape, z.shape)
print(x.min(), x.max())
print(y.min(), y.max())
print(z.min(), z.max())


plt.figure()
plt.subplot(131)
plt.imshow(x)
plt.subplot(132)
plt.imshow(y)
plt.subplot(133)
plt.imshow(z)
plt.show()

plt.figure()
plt.imshow(x)
plt.figure()
plt.imshow(y)
plt.figure()
plt.imshow(z)
plt.show()
