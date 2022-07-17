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

X = tb.imread('../../../data/images/Einstein256.png')

X = X
# X = X * 1e5

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

loss_cts_way1_fn = tb.ContrastLoss(mode='way1', axis=axis, reduction='mean')
loss_cts_way2_fn = tb.ContrastLoss(mode='way2', axis=axis, reduction='mean')

print(loss_mse_fn(Y.abs(), X.abs()))
print(loss_mae_fn(Y.abs(), X.abs()))
print(loss_cts_way1_fn(X))
print(loss_cts_way2_fn(X))
print(loss_cts_way1_fn(Y))
print(loss_cts_way2_fn(Y))

plt.figure()
plt.subplot(121)
plt.imshow(X.abs())
plt.subplot(122)
plt.imshow(Y.abs())
plt.show()
