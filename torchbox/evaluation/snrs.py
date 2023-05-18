#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : snrs.py
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
from torchbox import peakvalue, mse


def snr(x, n=None, **kwargs):
    r"""computes signal-to-noise ratio

    .. math::
        {\rm SNR} = 10*{\rm log10}(\frac{P_x}{P_n})
    
    where, :math:`P_x, P_n` are the power summary of the signal and noise:

    .. math::
       P_x = \sum_{i=1}^N |x_i|^2 \\
       P_n = \sum_{i=1}^N |n_i|^2 
    
    ``snr(x, n)`` equals to matlab's ``snr(x, n)``

    Parameters
    ----------
    x : tensor
        The pure signal data.
    n : ndarray, tensor
        The noise data.
    cdim : None or int, optional
        If :attr:`x` and :attr:`n` are complex-valued but represented in real format, 
        :attr:`cdim` or :attr:`cdim` should be specified. If not, it's set to :obj:`None`, 
        which means :attr:`x` and :attr:`n` are real-valued or complex-valued in complex format.
    dim : int or None, optional
        Specifies the dimensions for computing SNR, if not specified, it's set to :obj:`None`, 
        which means all the dimensions.
    keepdim : int or None, optional
        keep the complex dimension? (False for default)
    reduction : str, optional
        The reduce operation in batch dimension. Supported are ``'mean'``, ``'sum'`` or :obj:`None`.
        If not specified, it is set to :obj:`None`.
    
    Returns
    -----------
      : scalar
        The SNRs.

    Examples
    ----------

    ::

        import torch as th
        import torchbox as tb
    
        tb.setseed(seed=2020, target='torch')
        x = 10 * th.randn(5, 2, 3, 4)
        n = th.randn(5, 2, 3, 4)
        snrv = snr(x, n, cdim=1, dim=(2, 3), keepdim=True)
        print(snrv)
        snrv = snr(x, n, cdim=1, dim=(2, 3), keepdim=True, reduction='mean')
        print(snrv)
        x = tb.r2c(x, cdim=1)
        n = tb.r2c(n, cdim=1)
        snrv = snr(x, n, cdim=None, dim=(1, 2), reduction='mean')
        print(snrv)
        
        ---output
        tensor([[17.5840],
                [20.6824],
                [20.5385],
                [18.3238],
                [19.4630]])
        tensor(19.3183)
        tensor(19.3183)
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
        dim = None

    if 'keepdim' in kwargs:
        keepdim = kwargs['keepdim']
    elif 'keepcaxis' in kwargs:
        keepdim = kwargs['keepcaxis']
    else:
        keepdim = False

    if 'reduction' in kwargs:
        reduction = kwargs['reduction']
    else:
        reduction = None
        
    dim = tuple(range(x.dim())) if dim is None else dim

    if th.is_complex(x):  # complex in complex
        Px = th.sum(x.real*x.real + x.imag*x.imag, dim=dim)
        Pn = th.sum(n.real**2 + n.imag**2, dim=dim)
    elif cdim is None:  # real
        Px = th.sum(x**2, dim=dim)
        Pn = th.sum(n**2, dim=dim)
    else: # complex in real
        Px = th.sum(x**2, dim=cdim, keepdim=True)
        Pn = th.sum(n**2, dim=cdim, keepdim=True)
        Px = th.sum(Px, dim=dim, keepdim=keepdim)
        Pn = th.sum(Pn, dim=dim, keepdim=keepdim)
    
    S = 10 * th.log10(Px / Pn)
    if reduction in ['sum', 'SUM']:
        return th.sum(S)
    if reduction in ['mean', 'MEAN']:
        return th.mean(S)
    return S


def psnr(P, G, vpeak=None, **kwargs):
    r"""Peak Signal-to-Noise Ratio

    The Peak Signal-to-Noise Ratio (PSNR) is expressed as

    .. math::
        {\rm psnrv} = 10 \log10(\frac{V_{\rm peak}^2}{\rm MSE})

    For float data, :math:`V_{\rm peak} = 1`;

    For interges, :math:`V_{\rm peak} = 2^{\rm nbits}`,
    e.g. uint8: 255, uint16: 65535 ...


    Parameters
    -----------
    P : array_like
        The data to be compared. For image, it's the reconstructed image.
    G : array_like
        Reference data array. For image, it's the original image.
    vpeak : float, int or None, optional
        The peak value. If None, computes automaticly.
    cdim : None or int, optional
        If :attr:`P` and :attr:`G` are complex-valued but represented in real format, 
        :attr:`cdim` or :attr:`cdim` should be specified. If not, it's set to :obj:`None`, 
        which means :attr:`P` and :attr:`G` are real-valued or complex-valued in complex format.
    keepdim : int or None, optional
        keep the complex dimension?
    dim : int or None, optional
        Specifies the dimensions for computing SNR, if not specified, it's set to :obj:`None`, 
        which means all the dimensions.
    reduction : str, optional
        The reduce operation in batch dimension. Supported are ``'mean'``, ``'sum'`` or :obj:`None`.
        If not specified, it is set to :obj:`None`.
    
    Returns
    -------
    psnrv : float
        Peak Signal to Noise Ratio value.

    Examples
    ---------

    ::

        import torch as th
        import torchbox as tb
    
        tb.setseed(seed=2020, target='torch')
        P = 255. * th.rand(5, 2, 3, 4)
        G = 255. * th.rand(5, 2, 3, 4)
        snrv = psnr(P, G, vpeak=None, cdim=1, dim=(2, 3), keepdim=True)
        print(snrv)
        snrv = psnr(P, G, vpeak=None, cdim=1, dim=(2, 3), keepdim=True, reduction='mean')
        print(snrv)
        P = tb.r2c(P, cdim=1, keepdim=False)
        G = tb.r2c(G, cdim=1, keepdim=False)
        snrv = psnr(P, G, vpeak=255, cdim=None, dim=(1, 2), reduction='mean')
        print(snrv)

        # ---output
        tensor([[4.4584],
                [5.0394],
                [5.1494],
                [3.6585],
                [4.6466]])
        tensor(4.5905)
        tensor(4.5905)

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
        dim = None

    if 'keepdim' in kwargs:
        keepdim = kwargs['keepdim']
    elif 'keepcaxis' in kwargs:
        keepdim = kwargs['keepcaxis']
    else:
        keepdim = None

    if 'reduction' in kwargs:
        reduction = kwargs['reduction']
    else:
        reduction = None
        
    dim = tuple(range(th.ndim(P))) if dim is None else dim

    if P.dtype != G.dtype:
        print("Warning: P(" + str(P.dtype) + ")and G(" + str(G.dtype) +
              ")have different type! PSNR may not right!")

    if vpeak is None:
        vpeak = peakvalue(G)

    msev = mse(P, G, cdim=cdim, dim=dim, keepdim=keepdim, norm=False, reduction=None)
    psnrv = 10 * th.log10((vpeak ** 2) / msev)

    if reduction in ['mean', 'MEAN']:
       psnrv = th.mean(psnrv)
    if reduction in ['sum', 'SUM']:
       psnrv = th.sum(psnrv)

    return psnrv


if __name__ == '__main__':

    import torchbox as tb

    tb.setseed(seed=2020, target='torch')
    x = 10 * th.randn(5, 2, 3, 4)
    n = th.randn(5, 2, 3, 4)
    snrv = snr(x, n, cdim=1, dim=(2, 3), keepdim=True)
    print(snrv)
    snrv = snr(x, n, cdim=1, dim=(2, 3), keepdim=True, reduction='mean')
    print(snrv)
    x = tb.r2c(x, cdim=1)
    n = tb.r2c(n, cdim=1)
    snrv = snr(x, n, cdim=None, dim=(1, 2), reduction='mean')
    print(snrv)

    print('---psnr')
    tb.setseed(seed=2020, target='torch')
    P = 255. * th.rand(5, 2, 3, 4)
    G = 255. * th.rand(5, 2, 3, 4)
    snrv = psnr(P, G, vpeak=None, cdim=1, dim=(2, 3), keepdim=True)
    print(snrv)
    snrv = psnr(P, G, vpeak=None, cdim=1, dim=(2, 3), keepdim=True, reduction='mean')
    print(snrv)
    P = tb.r2c(P, cdim=1, keepdim=False)
    G = tb.r2c(G, cdim=1, keepdim=False)
    snrv = psnr(P, G, vpeak=255, cdim=None, dim=(1, 2), reduction='mean')
    print(snrv)