#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : noising.py
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
import torchbox as tb


def awgns(x, snrv, **kwargs):
    """adds white gaussian noise to signal

    see `Adding noise with a desired signal-to-noise ratio <https://sites.ualberta.ca/~msacchi/SNR_Def.pdf>`_ .

    Parameters
    ----------
    x : Tensor
        The pure signal data.
    snrv : int, float list, ndarray or Tensor
        The signal-to-noise ratio values in dB. If it is a number, all the samples are noised with the sample level.
    cdim : None or int, optional
        If :attr:`x` is complex-valued but represented in real format, 
        :attr:`cdim` or :attr:`caxis` should be specified. If not, it's set to :obj:`None`, 
        which means :attr:`x` is real-valued or complex-valued in complex format.
    dim : int or None, optional
        Specifies the dimensions for adding noise, if not specified, it's set to :obj:`None`, 
        which means all the dimensions.
    seed : int or None, optional
        Specifies the seed for generating random noise, if not specified, it's set to :obj:`None`.
    retall : bool, optional
        If :obj:`True`, noise will also be returned.

    Returns
    -----------
    y : Tensor
        The noised tensor.
    
    see also :func:`awgns2`.

    Examples
    ---------

    ::

        import torch as th
        import torchbox as tb

        tb.setseed(2020)
        x = th.randn(5, 2, 30, 40)
        x = tb.r2c(x, cdim=1)  # 5, 30, 40
        y, n = awgns(x, 30, dim=(1, 2), seed=2022, retall=True)
        snrv = tb.snr(x, n, dim=(1, 2))
        print(snrv, 'complex-valued in complex-format')
        
        tb.setseed(2020)
        x = th.randn(5, 2, 30, 40)
        y, n = awgns(x, 30, cdim=1, dim=(2, 3), seed=2022, retall=True)
        snrv = tb.snr(x, n, cdim=1, dim=(2, 3))
        print(snrv, 'complex-valued in real-format')

        tb.setseed(2020)
        x = th.randn(5, 2, 30, 40)
        y, n = awgns(x, 30, cdim=None, dim=(1, 2, 3), seed=2022, retall=True)
        snrv = tb.snr(x, n, cdim=None, dim=(1, 2, 3))
        print(snrv, 'real-valued in real-format')

        tb.setseed(2020)
        x = th.randn(5, 2, 30, 40)
        y, n = awgns2(x, 30, cdim=1, dim=(2, 3), seed=2022, retall=True)
        snrv = tb.snr(x, n, cdim=1, dim=(2, 3))
        print(snrv, 'real-valued in real-format, multi-channel')

        # ---output
        tensor([30.0846, 30.0605, 29.9890, 30.0245, 30.0455]) complex-valued in complex-format
        tensor([30.0846, 30.0605, 29.9890, 30.0245, 30.0455]) complex-valued in real-format
        tensor([30.1311, 30.0225, 30.0763, 30.0549, 30.1034]) real-valued in real-format
        tensor([30.1033, 30.0459, 29.9889, 29.8461, 29.9115]) real-valued in real-format, multi-channel
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

    if 'seed' in kwargs:
        seed = kwargs['seed']
    else:
        seed = None

    if 'retall' in kwargs:
        retall = kwargs['retall']
    else:
        retall = None

    tb.setseed(seed=seed, target='torch')

    if type(snrv) is not th.Tensor:
        snrv = th.tensor(snrv)

    linearSNR = 10**(snrv / 10.)

    if (not th.is_complex(x)) and (cdim is not None):
        n = tb.c2r(th.randn_like(tb.r2c(x, cdim=cdim, keepdim=True)), cdim=cdim, keepdim=True)
    else:
        n = th.randn_like(x)

    Px = th.sum(tb.pow(x, cdim=cdim, keepdim=True), dim=dim, keepdim=True)
    Pn = th.sum(tb.pow(n, cdim=cdim, keepdim=True), dim=dim, keepdim=True)

    if linearSNR.numel() != 1:
        linearSNR = th.reshape(linearSNR, Pn.shape) 
    alpha = th.sqrt(Px / linearSNR / Pn)
    n = alpha * n
    y = x + n
    if retall:
        return y, n
    else:
        return y

def awgns2(x, snrv, **kwargs):
    """adds white gaussian noise to multi-channel signal

    see `Adding noise with a desired signal-to-noise ratio <https://sites.ualberta.ca/~msacchi/SNR_Def.pdf>`_ .

    Parameters
    ----------
    x : Tensor
        The pure real-valued multi-channel signal data.
    snrv : int or float
        The signal-to-noise ratio value in dB.
    cdim : None or int, optional
        Specifies the channel dimension. If not specified, :attr:`x` will be treated as
        single-channel signal.
    dim : int or None, optional
        Specifies the dimensions for adding noise, if not specified, it's set to :obj:`None`, 
        which means all the dimensions.
    seed : int or None, optional
        Specifies the seed for generating random noise, if not specified, it's set to :obj:`None`.
    retall : bool, optional
        If :obj:`True`, noise will also be returned.

    Returns
    -----------
    y : Tensor
        The SNRs.

    see also :func:`awgns`.

    Examples
    ---------

    .. image:: ./_static/DemoNoiseAWGNS.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

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

    if 'seed' in kwargs:
        seed = kwargs['seed']
    else:
        seed = None

    if 'retall' in kwargs:
        retall = kwargs['retall']
    else:
        retall = None

    tb.setseed(seed=seed, target='torch')

    linearSNR = 10**(snrv / 10.)

    dim = tuple(range(x.dim())) if dim is None else dim
    
    if cdim is None:  # single-channel
        n = th.randn_like(x)
        Px = th.sum(x**2, dim=dim, keepdim=True)
        Pn = th.sum(n**2, dim=dim, keepdim=True)
    else:  # multi-channel
        n = th.zeros_like(x)
        nc = x.shape[cdim]
        d = x.dim()
        for i in range(nc):
            index = tb.sl(d, cdim, idx=[i])
            n[index] = th.randn_like(x[index])
        Px = th.sum(x**2, dim=dim, keepdim=True)
        Pn = th.sum(n**2, dim=dim, keepdim=True)

    alpha = th.sqrt(Px / linearSNR / Pn)
    n = alpha * n
    y = x + n
    if retall:
        return y, n
    else:
        return y


def imnoise(x, noise='awgn', snrv=30, fmt='chnllast'):
    r"""Add noise to image

    Add noise to each channel of the image.

    Parameters
    ----------
    x : Tensor
        image aray
    noise : str, optional
        noise type (the default is 'awgn', which means white gaussian noise, using :func:`awgn`)
    snrv : float, optional
        Signal-to-noise ratio (the default is 30, which [default_description])
    peak : None, str or float
        Peak value in input, if None, auto detected (default), if ``'maxv'``, use the maximum value as peak value.
    fmt : str or None, optional
        for color image, :attr:`fmt` should be specified with ``'chnllast'`` or ``'chnlfirst'``, for gray image, :attr:`fmt` should be setted to :obj:`None`.

    Returns
    -------
    tensor
        Images with added noise.

    Examples
    ---------

    .. image:: ./_static/DemoIMNOISE.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        datafolder = tb.data_path('optical')
        xr = tb.imread(datafolder + 'Einstein256.png')
        xi = tb.imread(datafolder + 'LenaGRAY256.png')

        x = xr + 1j * xi

        xnp15 = tb.imnoise(x, 'awgn', snrv=15)
        xn0 = tb.imnoise(x, 'awgn', snrv=0)
        xnn5 = tb.imnoise(x, 'awgn', snrv=-5)

        x = tb.abs(x, cdim=None)
        xnp15 = tb.abs(xnp15, cdim=None)
        xn0 = tb.abs(xn0, cdim=None)
        xnn5 = tb.abs(xnn5, cdim=None)

        plt = tb.imshow([x, xnp15, xn0, xnn5], titles=['original', 'noised(15dB)', 'noised(0dB)', 'noised(-5dB)'])
        plt.show()


        datafolder = tb.data_path('optical')
        xr = tb.imread(datafolder + 'Einstein256.png')
        xi = tb.imread(datafolder + 'LenaGRAY256.png')

        x = xr + 1j * xi
        x = tb.c2r(x, cdim=-1)
        print(x.shape, x.max())

        xnp15 = tb.imnoise(x, 'awgn', snrv=15)
        xn0 = tb.imnoise(x, 'awgn', snrv=0)
        xnn5 = tb.imnoise(x, 'awgn', snrv=-5)

        x = tb.abs(x, cdim=-1)
        xnp15 = tb.abs(xnp15, cdim=-1)
        xn0 = tb.abs(xn0, cdim=-1)
        xnn5 = tb.abs(xnn5, cdim=-1)

        plt = tb.imshow([x, xnp15, xn0, xnn5], titles=['original', 'noised(15dB)', 'noised(0dB)', 'noised(-5dB)'])
        plt.show()


    """

    img = x.clone()

    if noise not in ['awgn', 'AWGN']:
        raise ValueError('Not supported noise: %s' % noise)
    if img.dim() == 2:
        img = awgn(img, snrv, pmode='db', power='measured')
    elif img.dim() == 3:
        if fmt in ['chnllast', 'ChnlLast']:
            for c in range(img.shape[-1]):
                img[..., c] = awgn(img[..., c], snrv, pmode='db', power='measured')
        if fmt in ['chnlfirst', 'ChnlFirst']:
            for c in range(img.shape[0]):
                img[c, ...] = awgn(img[c, ...], snrv, pmode='db', power='measured')
        if fmt is None:  # gray image
            for n in range(img.shape[0]):
                img[n, ...] = awgn(img[n, ...], snrv, pmode='db', power='measured')
    elif img.dim() == 4:
        if fmt in ['chnllast', 'ChnlLast']:
            for n in range(img.shape[0]):
                for c in range(img.shape[-1]):
                    img[n, :, : c] = awgn(img[n, :, : c], snrv, pmode='db', power='measured')
        if fmt in ['chnlfirst', 'ChnlFirst']:
            for n in range(img.shape[0]):
                for c in range(img.shape[1]):
                    img[n, c, ...] = awgn(img[n, c, ...], snrv, pmode='db', power='measured')
    return img


def awgn(sig, snrv=30, pmode='db', power='measured', seed=None, retall=False):
    r"""AWGN Add white Gaussian noise to a signal.

    AWGN Add white Gaussian noise to a signal like matlab.

    Y = AWGN(X,snrv) adds white Gaussian noise to X.  The snrv is in dB.
    The power of X is assumed to be 0 dBW.  If X is complex, then
    AWGN adds complex noise.

    Parameters
    ----------
    sig : Tensor
        Signal that will be noised.
    snrv : float, optional
        Signal Noise Ratio (the default is 30)
    pmode : str, optional
        Power mode ``'linear'``, ``'db'`` (the default is 'db')
    power : float, str, optional
        the power of signal or the method for computing power (the default is 'measured', which is sigPower = th.sum(th.abs(sig) ** 2) / sig.numel())
    seed : int, optional
        Seed for random number generator. (the default is None, which means different each time)
    retall : bool, optional
        If :obj:`True`, noise will also be returned.
    
    Returns
    -------
    tensor
        noised data

    Raises
    ------
    IOError
        No input signal
    TypeError
        Input signal shape wrong
    """

    # --- Set default values
    sigPower = 1.  # linear, default
    linearSNRv = 10**(snrv / 10)

    # --- Check the signal power.
    # This needs to consider power measurements on matrices
    if power == 'measured':
        sigPower = th.sum(th.abs(sig) ** 2) / sig.numel()
    elif pmode in ['db', 'dbw']:
        sigPower = 10**(power / 10)
    elif pmode in ['dbm']:
        sigPower = 10 ** ((power - 30) / 10)

    pmode = 'linear'
    # --- Compute the required noise power
    noisePower = sigPower / linearSNRv
    
    # --- Add the noise
    if (th.is_complex(sig)):
        dtype = 'complex'
    else:
        dtype = 'real'

    noise = wgn(sig.shape, noisePower, pmode, dtype, seed, device=sig.device)
    y = sig + noise
    if retall:
        return y, noise
    else:
        return y


def wgn(shape, power, pmode='dbw', dtype='real', seed=None, device='cpu'):
    r"""WGN Generates white Gaussian noise.

    WGN Generates white Gaussian noise like matlab.

    Y = WGN((M,N),P) generates an M-by-N matrix of white Gaussian noise. P
    specifies the power of the output noise in dBW. The unit of measure for
    the output of the wgn function is Volts. For power calculations, it is
    assumed that there is a load of 1 Ohm.

    Parameters
    ----------
    shape : tuple
        Shape of noising matrix
    power : float
        P specifies the power of the output noise in dBW.
    pmode : str, optional
        Power mode of the output noise (the default is 'dbw')
    dtype : str, optional
        data type, real or complex (the default is 'real', which means real-valued)
    seed : int, optional
        Seed for random number generator. (the default is None, which means different each time)
    device : str, optional
        The device

    Returns
    -------
    tensor
        Matrix of white Gaussian noise (real or complex).
    """

    imp = 1.
    # print(shape)
    if pmode == 'linear':
        noisePower = power
    elif pmode in ['dbw', 'db']:
        noisePower = 10 ** (power / 10)
    elif pmode == 'dbm':
        noisePower = 10 ** ((power - 30) / 10)

    # --- Generate the noise
    if seed is not None:
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    if dtype == 'complex':
        y = (th.sqrt(imp * noisePower / 2)) * (th.randn(shape) + 1j * th.randn(shape))
    else:
        y = (th.sqrt(imp * noisePower)) * th.randn(shape)
    return y.to(device)


if __name__ == '__main__':

    
    tb.setseed(2020)
    x = th.randn(5, 2, 30, 40)
    x = tb.r2c(x, cdim=1)  # 5, 30, 40
    y, n = awgns(x, 2, dim=(1, 2), seed=2022, retall=True)
    snrv = tb.snr(x, n, dim=(1, 2))
    print(snrv, 'complex-valued in complex-format')
    
    tb.setseed(2020)
    x = th.randn(5, 2, 30, 40)
    y, n = awgns(x, [30, 10, -5, 5, 0], cdim=1, dim=(2, 3), seed=2022, retall=True)
    snrv = tb.snr(x, n, cdim=1, dim=(2, 3))
    print(snrv, 'complex-valued in real-format')

    tb.setseed(2020)
    x = th.randn(5, 2, 30, 40)
    y, n = awgns(x, -5, cdim=None, dim=(1, 2, 3), seed=2022, retall=True)
    snrv = tb.snr(x, n, cdim=None, dim=(1, 2, 3))
    print(snrv, 'real-valued in real-format')

    tb.setseed(2020)
    x = th.randn(5, 2, 30, 40)
    y, n = awgns2(x, 30, cdim=1, dim=(2, 3), seed=2022, retall=True)
    snrv = tb.snr(x, n, cdim=1, dim=(2, 3))
    print(snrv, 'real-valued in real-format, multi-channel')

    datafolder = tb.data_path('optical')
    xr = tb.imread(datafolder + 'EinsteinGRAY256.png')
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