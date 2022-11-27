#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : fft_layers.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Nov 27 2019
# @version   : 0.0
# @license   : The Apache License 2.0
# @note      : 
# 
# The Apache 2.0 License
# Copyright (C) 2013- Zhi Liu
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
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
