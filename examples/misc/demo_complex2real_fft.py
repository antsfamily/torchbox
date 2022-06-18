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
xr = tb.imread(datafolder + 'Einstein256.png')
xi = tb.imread(datafolder + 'LenaGRAY256.png')

x = xr + 1j * xi

y = tb.ct2rt(x, axis=0)
z = tb.rt2ct(y, axis=0)

print(x.shape, y.shape, z.shape)
print(x.abs().min(), x.abs().max())
print(y.abs().min(), y.abs().max())
print(z.abs().min(), z.abs().max())


plt.figure()
plt.subplot(131)
plt.imshow(x.real)
plt.subplot(132)
plt.imshow(y.real)
plt.subplot(133)
plt.imshow(z.imag)
plt.show()

# plt.figure()
# plt.imshow(x)
# plt.figure()
# plt.imshow(y)
# plt.figure()
# plt.imshow(z)
# plt.show()
