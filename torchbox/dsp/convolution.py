#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : convolution.py
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

import numpy as np
import torch as th
from torchbox.dsp.ffts import padfft, fft, ifft
from torchbox.base.mathops import nextpow2, ematmul, r2c, c2r
from torchbox.base.arrayops import cut


def cutfftconv1(y, nfft, Nx, Nh, shape='same', dim=0, ftshift=False):
    r"""Throwaway boundary elements to get convolution results.

    Throwaway boundary elements to get convolution results.

    Parameters
    ----------
    y : Tensor
        array after ``iff``.
    nfft : int
        number of fft points.
    Nx : int
        signal length
    Nh : int
        filter length
    shape : str
        output shape:
        1. ``'same' --> same size as input x``, :math:`N_x`
        2. ``'valid' --> valid convolution output``
        3. ``'full' --> full convolution output``, :math:`N_x+N_h-1`
        (the default is 'same')
    dim : int
        convolution dimension (the default is 0)
    ftshift : bool
        whether to shift zero the frequency to center (the default is False)

    Returns
    -------
    y : Tensor
        array with shape specified by :attr:`same`.
    """

    nfft, Nx, Nh = np.int32([nfft, Nx, Nh])
    N = Nx + Nh - 1
    Nextra = nfft - N

    if nfft < N:
        raise ValueError("~~~To get right results, nfft must be larger than Nx+Nh-1!")

    if ftshift:
        if np.mod(Nx, 2) > 0 and np.mod(Nh, 2) > 0:
            if Nextra > 0:
                Nhead = np.int32(np.fix((Nextra + 1) / 2.))
                Ntail = Nextra - Nhead
                y = cut(y, ((Nhead, np.int32(nfft - Ntail)),), dim)
            else:
                y = cut(y, ((N - 1, N), (0, N - 1)), dim)
        else:
            Nhead = np.int32(np.fix(Nextra / 2.))
            Ntail = Nextra - Nhead
            y = cut(y, ((Nhead, np.int32(nfft - Ntail)),), dim)
    else:
        Nhead = 0
        Ntail = Nextra
        y = cut(y, ((Nhead, np.int32(nfft - Ntail)),), dim)

    if shape in ['same', 'SAME', 'Same']:
        Nstart = np.fix(Nh / 2.)
        Nend = Nstart + Nx
    elif shape in ['valid', 'VALID', 'Valid']:
        Nstart = Nh - 1
        Nend = N - (Nh - 1)
    elif shape in ['full', 'FULL', 'Full']:
        Nstart, Nend = (0, N)
    Nstart, Nend = np.int32([Nstart, Nend])
    y = cut(y, ((Nstart, Nend),), dim)
    return y


def fftconv1(x, h, shape='same', nfft=None, ftshift=False, eps=None, **kwargs):
    r"""Convolution using Fast Fourier Transformation

    Convolution using Fast Fourier Transformation.

    Parameters
    ----------
    x : Tensor
        data to be convolved.
    h : Tensor
        filter array
    shape : str, optional
        output shape:
        1. ``'same' --> same size as input x``, :math:`N_x`
        2. ``'valid' --> valid convolution output``
        3. ``'full' --> full convolution output``, :math:`N_x+N_h-1`
        (the default is 'same')
    cdim : int or None
        If :attr:`x` is complex-valued, :attr:`cdim` is ignored. If :attr:`x` is real-valued and :attr:`cdim` is integer
        then :attr:`x` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex dim;
        otherwise (None), :attr:`x` will be treated as real-valued.
    dim : int, optional
        axis of fft operation (the default is 0, which means the first dimension)
    nfft : int, optional
        number of fft points (the default is :math:`2^{nextpow2(N_x+N_h-1)}`),
        note that :attr:`nfft` can not be smaller than :math:`N_x+N_h-1`.
    ftshift : bool, optional
        whether shift frequencies (the default is False)
    eps : None or float, optional
        x[abs(x)<eps] = 0 (the default is None, does nothing)

    Returns
    -------
    y : Tensor
        Convolution result array.

    """

    if 'cdim' in kwargs:
        cdim = kwargs['cdim']
    elif 'caxis' in kwargs:
        cdim = kwargs['caxis']
    else:
        cdim = None

    if 'dim' in kwargs:
        dim = kwargs['dim']
    elif 'axis' in kwargs:
        dim = kwargs['axis']
    else:
        dim = 0

    CplxRealflag = False
    if (not th.is_complex(x)) and (cdim is not None):  # complex in real
        CplxRealflag = True
        x = r2c(x, cdim=cdim, keepdim=True)
        h = r2c(h, cdim=cdim, keepdim=True)

    dh, dx = h.dim(), x.dim()
    if dh != dx:
        size = [1] * dx
        size[-1] = 2
        size[dim] = int(h.numel() / 2.)
        h = h.reshape(size)

    Nh = h.size(dim)
    Nx = x.size(dim)
    N = Nx + Nh - 1
    if nfft is None:
        nfft = 2**nextpow2(N)
    else:
        if nfft < N:
            raise ValueError("~~~To get right results, nfft must be larger than Nx+Nh-1!")

    x = padfft(x, nfft, dim, ftshift)
    h = padfft(h, nfft, dim, ftshift)
    x = fft(x, nfft, cdim=None, dim=dim, keepdim=False, norm=None, shift=ftshift)
    h = fft(h, nfft, cdim=None, dim=dim, keepdim=False, norm=None, shift=ftshift)
    y = ematmul(x, h)  # element-by-element complex multiplication

    y = ifft(y, nfft, cdim=None, dim=dim, keepdim=False, norm=None, shift=ftshift)
    y = cutfftconv1(y, nfft, Nx, Nh, shape, dim, ftshift)

    if eps is not None:
        y[abs(y) < eps] = 0.

    if CplxRealflag:
        y = c2r(y, cdim=cdim, keepdim=True)

    return y


if __name__ == '__main__':
    import pyaibox as pb
    import torchbox as tb
    import torch as th

    shape = 'same'
    ftshift = False
    ftshift = True
    x_np = np.array([1, 2, 3 + 6j, 4, 5])
    h_np = np.array([1 + 2j, 2, 3, 4, 5, 6, 7])

    x_th = th.from_numpy(x_np)
    h_th = th.from_numpy(h_np)

    y1 = pb.fftconv1(x_np, h_np, axis=0, nfft=None, shape=shape, ftshift=ftshift)
    y2 = tb.fftconv1(x_th, h_th, axis=0, nfft=None, shape=shape, ftshift=ftshift)

    y2 = y2.cpu().numpy()

    print(y1)
    print(y2)
    print(np.sum(np.abs(y1) - np.abs(y2)), np.sum(np.angle(y1) - np.angle(y2)))


    shape = 'same'
    ftshift = False
    ftshift = True
    x_np = np.array([1, 2, 3 + 6j, 4, 5])
    h_np = np.array([1 + 2j, 2, 3, 4, 5, 6, 7])
    x_th = th.from_numpy(x_np)
    h_th = th.from_numpy(h_np)

    x_np = pb.c2r(x_np, caxis=-1)
    h_np = pb.c2r(h_np, caxis=-1)
    x_th = tb.c2r(x_th, cdim=-1)
    h_th = tb.c2r(h_th, cdim=-1)

    y1 = pb.fftconv1(x_np, h_np, caxis=-1, axis=0, nfft=None, shape=shape, ftshift=ftshift)
    y2 = tb.fftconv1(x_th, h_th, caxis=-1, axis=0, nfft=None, shape=shape, ftshift=ftshift)

    y2 = y2.cpu().numpy()

    print(y1)
    print(y2)
    print(np.sum(np.abs(y1) - np.abs(y2)), np.sum(np.angle(y1) - np.angle(y2)))
