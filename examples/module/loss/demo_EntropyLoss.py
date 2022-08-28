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
dim = (0, 1)

X = tb.imread('data/images/oi/Einstein256.png')

X = X
# X = X * 1e5

X = X + 1j * X

print(X.shape)

E = th.exp(1j * th.randn(X.shape))

Y = X.clone()
for a in dim:
    Y = th.fft.fft(Y, dim=a)
Y = Y * E
for a in dim:
    Y = th.fft.ifft(Y, dim=a)

loss_mse_fn = th.nn.MSELoss(reduction='mean')
loss_mae_fn = th.nn.L1Loss(reduction='mean')

loss_ent_nat_fn = tb.EntropyLoss(mode='natural', cdim=None, dim=dim, reduction='mean')
loss_ent_sha_fn = tb.EntropyLoss(mode='shannon', cdim=None, dim=dim, reduction='mean')


print(loss_mse_fn(Y.abs(), X.abs()))
print(loss_mae_fn(Y.abs(), X.abs()))
print(loss_ent_nat_fn(X))
print(loss_ent_sha_fn(X))
print(loss_ent_nat_fn(Y))
print(loss_ent_sha_fn(Y))

plt.figure()
plt.subplot(121)
plt.imshow(X.abs())
plt.subplot(122)
plt.imshow(Y.abs())
plt.show()
