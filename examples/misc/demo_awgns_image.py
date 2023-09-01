#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-06-13 22:38:13
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchbox as tb

datafolder = tb.data_path('optical')
xr = tb.imread(datafolder + 'Einstein256.png')
xi = tb.imread(datafolder + 'LenaGRAY256.png')

xr = th.zeros(128, 128)
xi = th.zeros(128, 128)
x = xr + 1j * xi
print(x.shape)
x[64, 64] = 100 + 1j * 3
x[64, 32] = 20 + 1j * 8
x[32, 32] = 1 + 1j * 30

xnp15, np15 = tb.awgns(x, snrv=15, dim=(0, 1), retall=True)
xn0, n0 = tb.awgns(x, snrv=0, dim=(0, 1), retall=True)
xnn5, nn5 = tb.awgns(x, snrv=-5, dim=(0, 1), retall=True)

print(tb.snr(x, np15))
print(tb.snr(x, n0))
print(tb.snr(x, nn5))

x = tb.abs(x)
xnp15 = tb.abs(xnp15)
xn0 = tb.abs(xn0)
xnn5 = tb.abs(xnn5)

plt = tb.imshow([x, xnp15, xn0, xnn5], titles=['original', 'noised(15dB)', 'noised(0dB)', 'noised(-5dB)'])
plt.show()


datafolder = tb.data_path('optical')
xr = tb.imread(datafolder + 'Einstein256.png')
xi = tb.imread(datafolder + 'LenaGRAY256.png')

x = xr + 1j * xi
x = tb.c2r(x, cdim=-1)
print(x.shape)

xnp15, np15 = tb.awgns2(x, snrv=15, cdim=-1, dim=(0, 1), retall=True)
xn0, n0 = tb.awgns2(x, snrv=0, cdim=-1, dim=(0, 1), retall=True)
xnn5, nn5 = tb.awgns2(x, snrv=-5, cdim=-1, dim=(0, 1), retall=True)

print(tb.snr(x, np15, cdim=-1, dim=(0, 1)))
print(tb.snr(x, n0, cdim=-1, dim=(0, 1)))
print(tb.snr(x, nn5, cdim=-1, dim=(0, 1)))

x = tb.abs(x, cdim=-1)
xnp15 = tb.abs(xnp15, cdim=-1)
xn0 = tb.abs(xn0, cdim=-1)
xnn5 = tb.abs(xnn5, cdim=-1)

plt = tb.imshow([x, xnp15, xn0, xnn5], titles=['original', 'noised(15dB)', 'noised(0dB)', 'noised(-5dB)'])
plt.show()
