#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : fft_layers.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Nov 27 2019
# @version   : 0.0
# @license   : The GNU General Public License (GPL) v3.0
# @note      : 
# 
# The GNU General Public License (GPL) v3.0
# Copyright (C) 2013- Zhi Liu
#
# This file is part of torchbox.
#
# torchbox is free software: you can redistribute it and/or modify it under the 
# terms of the GNU General Public License as published by the Free Software Foundation, 
# either version 3 of the License, or (at your option) any later version.
#
# torchbox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with torchbox. 
# If not, see <https://www.gnu.org/licenses/>. 
#

import torch as th
import torch.nn.functional as F
import torchbox as tb


class FFTLayer1d(th.nn.Module):

    def __init__(self, nfft=None):
        super(FFTLayer1d, self).__init__()
        self.nfft = nfft

    def forward(self, x):
        n, d, _ = x.size()
        if self.nfft is None:
            self.nfft = d
        if d != self.nfft:
            x = F.pad(x, [0, self.nfft - d, 0], mode='constant', value=0)
        # y = th.fft.fft(x, n=None, dim=0, norm=None)
        y = tb.fft(x, n, cdim=-1, dim=0, norm=None)

        return y


if __name__ == '__main__':

    import numpy as np
    import torch as th
    import matplotlib.pyplot as plt

    PI = np.pi
    f0 = 100
    Fs = 1000
    Ts = 0.1
    Ns = int(Fs * Ts)

    f = np.linspace(0., Fs, Ns)
    t = np.linspace(0, Ts, Ns)
    x_np = np.cos(2. * PI * f0 * t) + 1j * np.sin(2. * PI * f0 * t)

    device = th.device('cuda:0')
    # x_th = th.tensor(x_np, dtype=th.complex64)
    x_th = th.tensor([x_np.real, x_np.imag], dtype=th.float32).transpose(1, 0)
    x_th = x_th.to(device)
    print(x_th.shape, type(x_th))

    x_ths = th.tensor([x_th.cpu().numpy(), x_th.cpu().numpy(),
                       x_th.cpu().numpy()], dtype=th.float32)

    print(x_ths.shape)

    fftlayer = FFTLayer1d()
    ys = fftlayer.forward(x_ths)
    ys = th.abs(ys[:, :, 0] + 1j * ys[:, :, 1]).cpu()

    plt.figure()
    plt.subplot(131)
    plt.plot(f, ys[0])
    plt.grid()
    plt.subplot(132)
    plt.plot(f, ys[1])
    plt.grid()
    plt.subplot(133)
    plt.plot(f, ys[2])
    plt.grid()
    plt.show()
