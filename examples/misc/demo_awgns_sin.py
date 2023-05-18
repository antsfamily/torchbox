#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2022-07-03 16:38:13
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchbox as tb

snrv = 15
fr = 100
fi = 200
Fs = 1000
Ts = 1.
Ns = int(Fs * Ts)
t = th.linspace(0, Ts, Ns)
xr = th.cos(2*th.pi*fr*t)
xi = th.cos(2*th.pi*fr*t)

x = xr + 1j * xi
print(x.shape)

xn1, n1 = tb.awgn(x, snrv=snrv, retall=True)
xn2, n2 = tb.awgns(x, snrv=snrv, retall=True)

snrv1 = tb.snr(x, n1)
snrv2 = tb.snr(x, n2)


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
