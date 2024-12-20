#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : transform.py
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
import torchbox as tb


def zscore(X, meanv=None, stdv=None, cdim=None, dim=None, retall=False):
    r"""standardization/zscore

    .. math::
        \bar{X} = \frac{X-\mu}{\sigma}


    Parameters
    ----------
    X : Tensor
        data to be normalized,
    meanv : list or None, optional
        mean value (the default is None, which means auto computed)
    stdv : list or None, optional
        standard deviation (the default is None, which means auto computed)
    cdim : int or None, optional
        complex dimension
    dim : list or int, optional
        specify the axis for computing mean and standard deviation (the default is None, which means all elements)
    retall : bool, optional
        if True, also return the mean and std (the default is False, which means just return the standardized data)
    """

    if type(X) is not th.Tensor:
        X = th.from_numpy(X)

    if meanv is None:
        meanv = tb.mean(X, cdim=cdim, dim=dim, keepdim=True)
    if stdv is None:
        stdv = tb.std(X, cdim=cdim, dim=dim, keepdim=True)
    if retall:
        return (X - meanv) / (stdv + tb.EPS), meanv, stdv
    else:
        return (X - meanv) / (stdv + tb.EPS)


def scale(X, st=[0, 1], sf=None, istrunc=True, retall=False):
    r"""
    Scale data.

    .. math::
        x \in [a, b] \rightarrow y \in [c, d]

    .. math::
        y = (d-c)(x-a) / (b-a) + c.

    Parameters
    ----------
    X : tensor_like
        The data to be scaled.
    st : tuple, list, optional
        Specifies the range of data after beening scaled. Default [0, 1].
    sf : tuple, list, optional
        Specifies the range of data. Default [min(X), max(X)].
    istrunc : bool
        Specifies wether to truncate the data to [a, b], For example,
        If sf == [a, b] and 'istrunc' is true,
        then X[X < a] == a and X[X > b] == b.
    retall : bool
        If ``True``, also return :attr:`st` and :attr:`sf`.

    Returns
    -------
    out : Tensor
        Scaled data tensor.
    st, sf : list or tuple
        If :attr:`retall` is true, also be returned
    """

    if type(X) is not th.Tensor:
        X = th.from_numpy(X)

    if X.dtype in tb.dtypes('int') + tb.dtypes('uint'):
        X = X.to(th.float64)

    if not(isinstance(st, (tuple, list)) and len(st) == 2):
        raise Exception("'st' is a tuple or list, such as (-1,1)")
    if sf is not None:
        if not(isinstance(sf, (tuple, list)) and len(sf) == 2):
            raise Exception("'sf' is a tuple or list, such as (0, 255)")
    else:
        sf = [th.min(X) + 0.0, th.max(X) + 0.0]
    if sf[0] is None:
        sf = (th.min(X) + 0.0, sf[1])
    if sf[1] is None:
        sf = (sf[0], th.max(X) + 0.0)

    a = sf[0] + 0.0
    b = sf[1] + 0.0
    c = st[0] + 0.0
    d = st[1] + 0.0

    if istrunc:
        X[X < a] = a
        X[X > b] = b

    if retall:
        return (X - a) * (d - c) / (b - a + tb.EPS) + c, st, sf
    else:
        return (X - a) * (d - c) / (b - a + tb.EPS) + c


def quantization(X, idrange=None, odrange=[0, 31], odtype='auto', retall=False):
    r"""

    Quantize data.

    .. math::
        x \in [a, b] \rightarrow y \in [c, d]

    .. math::
        y = (d-c)(x-a) / (b-a) + c.

    Parameters
    ----------
    X : Tensor
        The data to be quantized with shape :math:`N_a×N_r ∈ {\mathbb R}`, or :math:`N_a×N_r ∈ {\mathbb C}`.
    idrange : tuple, list, optional
        Specifies the range of data. Default [min(X), max(X)].
    odrange : tuple, list, optional
        Specifies the range of data after beening quantized. Default [0, 31].
    odtype : str, None, optional
        output data type, supportted are ``'auto'`` (auto infer, default), or torch tensor's dtype string.
        If the type of :attr:`odtype` is not string(such as None),
        the type of output data is the same with input.
    retall : bool
        If ``True``, also return :attr:`st` and :attr:`idrange`.

    Returns
    -------
    out : Tensor
        Quantized data tensor, if the input is complex, will return a tensor with shape :math:`N_a×N_r×2 ∈ {\mathbb R}`.
    idrange, odrange : list or tuple
        If :attr:`retall` is true, also be returned
    """

    if type(X) is not th.Tensor:
        X = th.from_numpy(X)

    if th.is_complex(X):
        X = th.view_as_real(X)

    if not(isinstance(odrange, (tuple, list)) and len(odrange) == 2):
        raise Exception("'st' is a tuple or list, such as (-1,1)")
    if idrange is not None:
        if not(isinstance(idrange, (tuple, list)) and len(idrange) == 2):
            raise Exception("'sf' is a tuple or list, such as (0, 255)")
    else:
        idrange = [X.min() + 0.0, X.max() + 0.0]
    if idrange[0] is None:
        idrange = (X.min() + 0.0, idrange[1])
    if idrange[1] is None:
        idrange = (idrange[0], X.max() + 0.0)

    if odtype in ['auto', 'AUTO']:
        if odrange[0] >= 0:
            odtype = 'th.uint'
        else:
            odtype = 'th.int'
        odtype = odtype + str(tb.nextpow2(odrange[1] - odrange[0]))

    if type(odtype) is str:
        X = X.to(eval(odtype))

    if retall:
        return X, idrange, odrange
    else:
        return X


def db20(x):
    r"""Computes dB value of a tensor

    Parameters
    ----------
    x : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        The output tensor (dB)
    """
    return 20. * th.log10(th.abs(x))


def ct2rt(x, dim=0):
    r"""Converts a complex-valued tensor to a real-valued tensor

    Converts a complex-valued tensor :math:`{\bf x}` to a real-valued tensor with FFT and conjugate symmetry.


    Parameters
    ----------
    x : Tensor
        The input tensor :math:`{\bf x}`.
    dim : int
        The axis for excuting FFT.

    Returns
    -------
    Tensor
        The output tensor :math:`{\bf y}`.

    see also :func:`rt2ct`.

    Examples
    ---------

    .. image:: ./_static/CT2RTRT2CTdemo.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::


        import torchbox as tb

        datafolder = tb.data_path('optical')
        xr = tb.imread(datafolder + 'Einstein256.png')
        xi = tb.imread(datafolder + 'LenaGRAY256.png')

        x = xr + 1j * xi

        y = tb.ct2rt(x, dim=0)
        z = tb.rt2ct(y, dim=0)

        print(x.shape, y.shape, z.shape)
        print(x.dtype, y.dtype, z.dtype)
        print(x.abs().min(), x.abs().max())
        print(y.abs().min(), y.abs().max())
        print(z.abs().min(), z.abs().max())


        plt = tb.imshow([x.real, x.imag, y.real, y.imag, z.real, z.imag], nrows=3, ncols=2,
                        titles=['original(real)', 'original(imag)', 'converted(real)', 
                        'converted(imag)', 'reconstructed(real)', 'reconstructed(imag)'])
        plt.show()

    """

    d = x.dim()
    n = x.shape[dim]
    X = th.fft.fft(x, dim=dim)
    X0 = X[tb.sl(d, dim, [[0]])]
    Y = th.cat((X0.real, X[tb.sl(d, dim, range(1, n))], X0.imag, th.conj(X[tb.sl(d, dim, range(n - 1, 0, -1))])), dim=dim)

    del x, X
    y = th.fft.ifft(Y, dim=dim)

    return y


def rt2ct(y, dim=0):
    r"""Converts a real-valued tensor to a complex-valued tensor

    Converts a real-valued tensor :math:`{\bf y}` to a complex-valued tensor with FFT and conjugate symmetry.


    Parameters
    ----------
    y : Tensor
        The input tensor :math:`{\bf y}`.
    dim : int
        The axis for excuting FFT.

    Returns
    -------
    Tensor
        The output tensor :math:`{\bf x}`.
    
    see also :func:`ct2rt`.

    """

    d = y.dim()
    n = y.shape[dim]

    Y = th.fft.fft(y, dim=dim)
    X = Y[tb.sl(d, dim, range(0, int(n / 2)))]
    X[tb.sl(d, dim, [[0]])] = X[tb.sl(d, dim, [[0]])] + 1j * Y[tb.sl(d, dim, [[int(n / 2)]])].real
    del y, Y
    x = th.fft.ifft(X, dim=dim)
    return x


if __name__ == '__main__':

    X = th.randn(4, 3, 5, 6)
    # X = th.randn(3, 4)
    XX = zscore(X, dim=(0, 2, 3))
    XX, meanv, stdv = zscore(X, dim=(0, 2, 3), retall=True)
    print(XX.size())
    print(meanv, stdv)

    X = np.random.randn(4, 3, 5, 6) * 255
    # X = th.randn(3, 4)
    XX = zscore(X, dim=(0, 2, 3))
    XX, meanv, stdv = zscore(X, dim=(0, 2, 3), retall=True)
    print(XX.size())
    print(meanv, stdv)
    print(XX)

    XX = scale(X, st=[0, 1])
    print(XX)


    x = th.tensor([1, 2, 3]) + th.tensor([1, 2, 3]) * 1j

    y = ct2rt(x)
    z = rt2ct(y)

    print(x)
    print(y)
    print(z)