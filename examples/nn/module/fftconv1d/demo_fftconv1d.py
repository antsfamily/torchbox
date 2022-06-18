#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-11-07 17:00:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torch as th
import torchbox as tb
import pyaibox as pb
import matplotlib.pyplot as plt

X_np = pb.imread('../../../../data/images/Einstein.png').astype('float32')
X_np = X_np + 1j * X_np
h_np = np.array([[0, -1, 1, 0]])
Y1 = X_np
Y1 = pb.fftconv1(Y1, h_np, axis=1, shape='same')
Y1 = pb.fftconv1(Y1, h_np.transpose(), axis=0, shape='same')
print(X_np.shape, h_np.shape, Y1.shape)

device = th.device('cuda:0')
X_th = th.tensor([X_np.real, X_np.imag], device=device, dtype=th.float32).permute(1, 2, 0)
h_th = th.tensor([[0, -1, 1, 0]], device=device, dtype=th.float32)
h_th = th.stack([h_th, th.zeros(h_th.size(), device=device, dtype=th.float32)], axis=-1)
Y2 = X_th
Y2 = tb.fftconv1(Y2, h_th, axis=1, shape='same')
Y2 = tb.fftconv1(Y2, h_th.transpose(0, 1), axis=0, shape='same')
Y2 = th.view_as_complex(Y2)
print(X_th.shape, h_th.shape, Y2.shape)

Y1 = np.abs(Y1)
Y2 = th.abs(Y2)

# Y1 = np.log10(Y1)
# Y2 = th.log10(Y2)

Y2 = Y2.cpu().numpy()
print(np.sum(Y1 - Y2))

plt.figure()
plt.subplot(131)
plt.imshow(np.abs(X_np))
plt.subplot(132)
plt.imshow(Y1)
plt.subplot(133)
plt.imshow(Y2)
plt.show()
