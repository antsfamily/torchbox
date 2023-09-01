#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : correlation.py
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
from torchbox.dsp.ffts import fft, ifft, padfft
from torchbox.base.mathops import nextpow2, ematmul, conj
from torchbox.base.arrayops import cut


def cutfftcorr1(y, nfft, Nx, Nh, shape='same', dim=0, ftshift=False):
    r"""Throwaway boundary elements to get correlation results.

    Throwaway boundary elements to get correlation results.

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
    shape : dstr
        output shape:
        1. ``'same' --> same size as input x``, :math:`N_x`
        2. ``'valid' --> valid correlation output``
        3. ``'full' --> full correlation output``, :math:`N_x+N_h-1`
        (the default is 'same')
    dim : int
        correlation dim (the default is 0)
    ftshift : bool
        whether to shift the frequencies (the default is False)

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
        if Nextra == 0:
            if np.mod(Nx, 2) == 0 and np.mod(Nh, 2) > 0:
                y = cut(y, ((0, N), ), dim)
            else:
                y = cut(y, ((1, nfft), (0, 1)), dim)
        else:
            if np.mod(Nx, 2) == 0 and np.mod(Nextra, 2) == 0:
                Nhead = np.int32(np.fix(Nextra / 2.))
            else:
                Nhead = np.int32(np.fix(Nextra / 2.) + 1)
            Nstart2 = Nhead
            Nend2 = np.int32(Nstart2 + N)
            y = cut(y, ((Nstart2, Nend2), ), dim)
    else:
        Nstart2 = 0
        Nend2 = Nx
        Nend1 = nfft
        Nstart1 = int(np.uint(Nend1 - (Nh - 1)))
        y = cut(y, ((Nstart1, Nend1), (Nstart2, Nend2)), dim)

    if shape in ['same', 'SAME', 'Same']:
        Nstart = np.uint(np.fix(Nh / 2.))
        Nend = np.uint(Nstart + Nx)
    elif shape in ['valid', 'VALID', 'Valid']:
        Nstart = np.uint(Nh - 1)
        Nend = np.uint(N - (Nh - 1))
    elif shape in ['full', 'FULL', 'Full']:
        Nstart, Nend = (0, N)
    y = cut(y, ((Nstart, Nend),), dim)
    return y


def fftcorr1(x, h, shape='same', nfft=None, ftshift=False, eps=None, **kwargs):
    """Correlation using Fast Fourier Transformation

    Correlation using Fast Fourier Transformation.

    Parameters
    ----------
    x : Tensor
        data to be convolved.
    h : Tensor
        filter array, it will be expanded to the same dimensions of :attr:`x` first.
    shape : dstr, optional
        output shape:
        1. ``'same' --> same size as input x``, :math:`N_x`
        2. ``'valid' --> valid correlation output``
        3. ``'full' --> full correlation output``, :math:`N_x+N_h-1`
        (the default is 'same')
    cdim : int or None
        If :attr:`x` is complex-valued, :attr:`cdim` is ignored. If :attr:`x` is real-valued and :attr:`cdim` is integer
        then :attr:`x` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex dim;
        otherwise (None), :attr:`x` will be treated as real-valued.
    dim : int, optional
        axis of fft operation (the default is 0, which means the first dimension)
    nfft : int, optional
        number of fft points (the default is None, :math:`2^{nextpow2(N_x+N_h-1)}`),
        note that :attr:`nfft` can not be smaller than :math:`N_x+N_h-1`.
    ftshift : bool, optional
        whether shift frequencies (the default is False)
    eps : None or float, optional
        x[abs(x)<eps] = 0 (the default is None, does nothing)

    Returns
    -------
    y : Tensor
        Correlation result array.

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
        x = tb.r2c(x, cdim=cdim, keepdim=True)
        h = tb.r2c(h, cdim=cdim, keepdim=True)

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
    x = fft(x, nfft, cdim=cdim, dim=dim, keepdim=False, norm=None, shift=ftshift)
    h = fft(h, nfft, cdim=cdim, dim=dim, keepdim=False, norm=None, shift=ftshift)
    h = conj(h, cdim=cdim)
    y = ematmul(x, h, cdim=cdim)  # element-by-element complex multiplication

    y = ifft(y, nfft, cdim=cdim, dim=dim, keepdim=False, norm=None, shift=ftshift)
    y = cutfftcorr1(y, nfft, Nx, Nh, shape, dim, ftshift)

    if eps is not None:
        y[abs(y) < eps] = 0.

    if CplxRealflag:
        y = tb.c2r(y, cdim=cdim, keepdim=True)

    return y


def xcorr(A, B, shape='same', dim=0):
    r"""Cross-correlation function estimates.


    Parameters
    ----------
    A : numpy array
        data1
    B : numpy array
        data2
    mod : str, optional
        - 'biased': scales the raw cross-correlation by 1/M.
        - 'unbiased': scales the raw correlation by 1/(M-abs(lags)).
        - 'coeff': normalizes the sequence so that the auto-correlations
                   at zero lag are identically 1.0.
        - 'none': no scaling (this is the default).
    """

    if np.ndim(A) == 1 and np.ndim(B) == 1:
        Ma, Mb = (1, 1)
        Na, Nb = (len(A), len(B))
    if np.ndim(A) == 2 and np.ndim(B) == 2:
        print(A.shape, B.shape)
        Ma, Na = A.shape
        Mb, Nb = B.shape
        if dim == 1 and Ma != Mb:
            raise ValueError("~~~Array A and B should have the same rows!")
        if dim == 0 and Na != Nb:
            raise ValueError("~~~Array A and B should have the same cols!")
    if shape in ['same', 'SAME']:
        Nc = max(Na, Nb)
    elif shape in ['full', 'FULL']:
        Nc = Na + Nb - 1
    elif shape in ['valid', 'VALID']:
        Nc = max(Na, Nb) - max(Na, Nb) + 1
    else:
        raise ValueError("~~~Not supported shape:" + shape + "!")

    CPLXDTYPESTR = ['complex128', 'complex64', 'complex']

    if A.dtype in CPLXDTYPESTR or B.dtype in CPLXDTYPESTR:
        dtype = 'complex'
    else:
        dtype = 'float'

    if np.ndim(A) == 1 and np.ndim(B) == 1:
        C = np.correlate(A, B, mode=shape)
    if np.ndim(A) == 2 and np.ndim(B) == 2:
        C = np.zeros((Ma, Nc), dtype=dtype)
        if dim == 0:
            for n in range(Na):
                C[:, n] = np.correlate(A[:, n], B[:, n], mode=shape)
        if dim == 1:
            for m in range(Ma):
                C[m, :] = np.correlate(A[m, :], B[m, :], mode=shape)
    return C


def acorr(x, P, dim=0, scale=None):
    r"""computes auto-correlation using fft

    Parameters
    ----------
    x : Tensor
        the input signal tensor
    P : int
        maxlag
    dim : int
        the auto-correlation dimension
    scale : str or None, optional
        :obj:`None`, ``'biased'`` or ``'unbiased'``, by default None
    """    

    M = x.shape[dim]
    mxl = min(P, M - 1)
    M2 = 2 * M

    dtype = 'complex' if th.is_complex(x) else 'real'

    x = th.fft.fft(x, n=M2, dim=dim)
    x = th.fft.ifft(x * x.conj(), dim=dim)  # output x is c

    x = cut(x, [(M2-mxl, M2), (0, mxl+1)], dim=dim)

    if dtype == 'real':
        x = x.real

    if scale == 'biased':
        x /= M
    if scale == 'unbiased':
        L = (x.shape[0] - 1) / 2
        s = M - th.arange(-L, L+1).abs()
        s[s<=0] = 1.
        sshape = [1] * x.ndim
        sshape[dim] = len(s)
        x /= s.reshape(sshape)

    return x


def accc(Sr, isplot=False):
    r"""Average cross correlation coefficient

    Average cross correlation coefficient (ACCC)

    .. math::
       \overline{C(\eta)}=\sum_{\eta} s^{*}(\eta) s(\eta+\Delta \eta)

    where, :math:`\eta, \Delta \eta` are azimuth time and it's increment.


    Parameters
    ----------
    Sr : numpy array
        SAR raw signal data :math:`N_a×N_r` or range compressed data.

    Returns
    -------
    1d array
        ACCC in each range cell.
    """

    Na, Nr = Sr.shape

    acccv = np.sum(Sr[1:, :] * np.conj(Sr[0:-1, :]), 0)

    if isplot:
        import matplotlib.pyplot as plt
        import torchbox
        plt.figure()
        plt.subplot(121)
        torchbox.cplot(acccv, '-b')
        plt.title('ACCC (all range cell)')
        plt.subplot(122)
        ccv = Sr[1:, 0] * np.conj(Sr[0:-1, 0])
        torchbox.cplot(ccv, '-b')
        torchbox.cplot([np.mean(ccv)], '-r')
        plt.title('CCC (0-th range cell)')
        plt.show()

    return acccv


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

    y1 = pb.fftcorr1(x_np, h_np, axis=0, nfft=None, shape=shape, ftshift=ftshift)
    y2 = tb.fftcorr1(x_th, h_th, dim=0, nfft=None, shape=shape, ftshift=ftshift)

    y2 = y2.cpu().numpy()

    print(y1)
    print(y2)
    print(np.sum(np.abs(y1) - np.abs(y2)), np.sum(np.angle(y1) - np.angle(y2)))


    x = th.tensor([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]]).T
    print(x, x.shape)

    c = acorr(x, 3, dim=0, scale=None)
    print(c, c.shape)

    c = acorr(x[:, 0:1], 3, dim=0, scale=None)
    print(c, c.shape)

    c = acorr(x[:, 0:1], 3, dim=0, scale='biased')
    print(c, c.shape)

    c = acorr(x[:, 0:1], 3, dim=0, scale='unbiased')
    print(c, c.shape)

    c = acorr(x[:, 0:1], 2, dim=0, scale='unbiased')
    print(c, c.shape)