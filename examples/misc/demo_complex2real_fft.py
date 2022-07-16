#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#
import torch as th
import torchbox as tb

datafolder = tb.data_path('optical')
xr = tb.imread(datafolder + 'Einstein256.png')
xi = tb.imread(datafolder + 'LenaGRAY256.png')

x = xr + 1j * xi

y = tb.ct2rt(x, dim=0)
z = tb.rt2ct(y, dim=0)

print(x.shape, y.shape, z.shape)
print(x.dtype, y.dtype, z.dtype)
print(x.abs().min(), x.abs().max())
print(y.abs().min(), y.abs().max())
print(z.abs().min(), z.abs().max())


plt = tb.imshow([x.real, x.imag, y.real, y.imag, z.real, z.imag], nrows=3, ncols=2,
                titles=['original(real)', 'original(imag)', 'converted(real)', 
                'converted(imag)', 'reconstructed(real)', 'reconstructed(imag)'])
plt.show()
