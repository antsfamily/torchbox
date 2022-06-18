#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-06-13 22:38:13
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torchbox as tb

datafolder = tb.data_path('shape')
x = tb.imread(datafolder + 'Airplane.png')

x = tb.rgb2gray(x)
print(x.shape, x.max())
xnp15 = tb.imnoise(x, 'awgn', snrv=15)
xn0 = tb.imnoise(x, 'awgn', snrv=0)
xnn5 = tb.imnoise(x, 'awgn', snrv=-5)

plt = tb.imshow([x, xnp15, xn0, xnn5], titles=['original', 'noised(15dB)', 'noised(0dB)', 'noised(-5dB)'])
plt.show()

datafolder = tb.data_path('optical')
xr = tb.imread(datafolder + 'Einstein256.png')
xi = tb.imread(datafolder + 'LenaGRAY256.png')

x = xr + 1j * xi

xnp15 = tb.imnoise(x, 'awgn', snrv=15)
xn0 = tb.imnoise(x, 'awgn', snrv=0)
xnn5 = tb.imnoise(x, 'awgn', snrv=-5)

x = tb.abs(x, cdim=None)
xnp15 = tb.abs(xnp15, cdim=None)
xn0 = tb.abs(xn0, cdim=None)
xnn5 = tb.abs(xnn5, cdim=None)

plt = tb.imshow([x, xnp15, xn0, xnn5], titles=['original', 'noised(15dB)', 'noised(0dB)', 'noised(-5dB)'])
plt.show()


datafolder = tb.data_path('optical')
xr = tb.imread(datafolder + 'Einstein256.png')
xi = tb.imread(datafolder + 'LenaGRAY256.png')

x = xr + 1j * xi
x = tb.c2r(x, cdim=-1)
print(x.shape, x.max())

xnp15 = tb.imnoise(x, 'awgn', snrv=15)
xn0 = tb.imnoise(x, 'awgn', snrv=0)
xnn5 = tb.imnoise(x, 'awgn', snrv=-5)

x = tb.abs(x, cdim=-1)
xnp15 = tb.abs(xnp15, cdim=-1)
xn0 = tb.abs(xn0, cdim=-1)
xnn5 = tb.abs(xnn5, cdim=-1)

plt = tb.imshow([x, xnp15, xn0, xnn5], titles=['original', 'noised(15dB)', 'noised(0dB)', 'noised(-5dB)'])
plt.show()
