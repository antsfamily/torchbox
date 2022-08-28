#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-18 10:14:12
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torchbox as tb
import matplotlib.pyplot as plt

device = 'cpu'
device = 'cuda:0'
axis = (0, 1)

X = tb.imread('data/images/oi/Einstein256.png')

X = X
X = X * 1e5

X = X + 1j * X

print(X.shape)

E = th.exp(1j * th.randn(X.shape))

Y = X.clone()
for a in axis:
    Y = th.fft.fft(Y, dim=a)
Y = Y * E
for a in axis:
    Y = th.fft.ifft(Y, dim=a)

loss_mse_fn = th.nn.MSELoss(reduction='mean')
loss_mae_fn = th.nn.L1Loss(reduction='mean')

loss_fda_mse_fn = tb.FourierLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
loss_fda_mae_fn = tb.FourierLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mae', reduction='mean')

print(loss_mse_fn(Y.abs(), X.abs()), "MSE")
print(loss_mae_fn(Y.abs(), X.abs()), "MAE")
print(loss_fda_mse_fn(Y, X), "FDA MSE")
print(loss_fda_mae_fn(Y, X), "FDA MAE")

th.manual_seed(2020)
xr = th.randn(10, 2, 4, 4) * 10000
yr = th.randn(10, 2, 4, 4) * 10000
xc = xr[:, [0], ...] + 1j * xr[:, [1], ...]
yc = yr[:, [0], ...] + 1j * yr[:, [1], ...]

flossr = tb.FourierLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
flossc = tb.FourierLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
print(flossr(xr, yr))
print(flossc(xc, yc))

flossr = tb.FourierLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
flossc = tb.FourierLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
print(flossr(xr, yr))
print(flossc(xc, yc))

plt.figure()
plt.subplot(121)
plt.imshow(X.abs())
plt.subplot(122)
plt.imshow(Y.abs())
plt.show()
