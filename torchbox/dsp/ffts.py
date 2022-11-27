#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : ffts.py
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

import numpy as np
import torch as th
import torch.fft as thfft
import torchbox as tb


def freq(n, fs, norm=False, shift=False, dtype=th.float32, device='cpu'):
    r"""Return the sample frequencies

    Return the sample frequencies.

    Given a window length `n` and a sample rate `fs`, if shift is ``True``::

      f = [-n/2, ..., n/2] / (d*n)

    Given a window length `n` and a sample rate `fs`, if shift is ``False``::

      f = [0, 1, ..., n] / (d*n)

    If :attr:`norm` is ``True``, :math:`d = 1`, else :math:`d = 1/f_s`.

    Parameters
    ----------
    fs : float
        Sampling rate.
    n : int
        Number of samples.
    norm : bool
        Normalize the frequencies.
    shift : bool
        Does shift the zero frequency to center.
    dtype : torch tensor type
        Data type, default is ``th.float32``.
    device : str
        device string, default is ``'cpu'``.

    Returns
    -------
    torch 1d-tensor
        Frequency array with size :math:`n×1`.

    Examples
    --------

    ::

        import torchbox as tb

        n = 10
        print(np.fft.fftfreq(n, d=0.1), 'numpy')
        print(th.fft.fftfreq(n, d=0.1), 'torch')
        print(tb.fftfreq(n, fs=10., norm=False), 'fftfreq, norm=False, shift=False')
        print(tb.fftfreq(n, fs=10., norm=True), 'fftfreq, norm=True, shift=False')
        print(tb.fftfreq(n, fs=10., shift=True), 'fftfreq, norm=False, shift=True')
        print(tb.freq(n, fs=10., norm=False), 'freq, norm=False, shift=False')
        print(tb.freq(n, fs=10., norm=True), 'freq, norm=True, shift=False')
        print(tb.freq(n, fs=10., shift=True), 'freq, norm=False, shift=True')

        # ---output
        [ 0.  1.  2.  3.  4. -5. -4. -3. -2. -1.] numpy
        tensor([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.]) torch
        tensor([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.]) fftfreq, norm=False, shift=False
        tensor([ 0.0000,  0.1000,  0.2000,  0.3000,  0.4000, -0.5000, -0.4000, -0.3000,
                -0.2000, -0.1000]) fftfreq, norm=True, shift=False
        tensor([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.]) fftfreq, norm=False, shift=True
        tensor([ 0.0000,  1.1111,  2.2222,  3.3333,  4.4444,  5.5556,  6.6667,  7.7778,
                8.8889, 10.0000]) freq, norm=False, shift=False
        tensor([0.0000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889,
                1.0000]) freq, norm=True, shift=False
        tensor([-5.0000, -3.8889, -2.7778, -1.6667, -0.5556,  0.5556,  1.6667,  2.7778,
                3.8889,  5.0000]) freq, norm=False, shift=True
    """

    d = 1. / fs

    if shift:
        f = th.linspace(-n / 2., n / 2., n, dtype=dtype, device=device)
    else:
        f = th.linspace(0, n, n, dtype=dtype, device=device)

    if norm:
        return f / n
    else:
        return f / (d * n)


def fftfreq(n, fs, norm=False, shift=False, dtype=th.float32, device='cpu'):
    r"""Return the Discrete Fourier Transform sample frequencies

    Return the Discrete Fourier Transform sample frequencies.

    Given a window length `n` and a sample rate `fs`, if shift is ``True``::

      f = [-n/2, ..., -1,     0, 1, ...,   n/2-1] / (d*n)   if n is even
      f = [-(n-1)/2, ..., -1, 0, 1, ..., (n-1)/2] / (d*n)   if n is odd

    Given a window length `n` and a sample rate `fs`, if shift is ``False``::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    If :attr:`norm` is ``True``, :math:`d = 1`, else :math:`d = 1/f_s`.

    Parameters
    ----------
    n : int
        Number of samples.
    fs : float
        Sampling rate.
    norm : bool
        Normalize the frequencies.
    shift : bool
        Does shift the zero frequency to center.
    dtype : torch tensor type
        Data type, default is ``th.float32``.
    device : torch device
        device string, ``'cpu', 'cuda:0', 'cuda:1'``

    Returns
    -------
    torch 1d-array
        Frequency array with size :math:`n×1`.

    Examples
    --------

    ::

        import torchbox as tb

        n = 10
        print(np.fft.fftfreq(n, d=0.1), 'numpy')
        print(th.fft.fftfreq(n, d=0.1), 'torch')
        print(tb.fftfreq(n, fs=10., norm=False), 'fftfreq, norm=False, shift=False')
        print(tb.fftfreq(n, fs=10., norm=True), 'fftfreq, norm=True, shift=False')
        print(tb.fftfreq(n, fs=10., shift=True), 'fftfreq, norm=False, shift=True')
        print(tb.freq(n, fs=10., norm=False), 'freq, norm=False, shift=False')
        print(tb.freq(n, fs=10., norm=True), 'freq, norm=True, shift=False')
        print(tb.freq(n, fs=10., shift=True), 'freq, norm=False, shift=True')

        # ---output
        [ 0.  1.  2.  3.  4. -5. -4. -3. -2. -1.] numpy
        tensor([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.]) torch
        tensor([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.]) fftfreq, norm=False, shift=False
        tensor([ 0.0000,  0.1000,  0.2000,  0.3000,  0.4000, -0.5000, -0.4000, -0.3000,
                -0.2000, -0.1000]) fftfreq, norm=True, shift=False
        tensor([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.]) fftfreq, norm=False, shift=True
        tensor([ 0.0000,  1.1111,  2.2222,  3.3333,  4.4444,  5.5556,  6.6667,  7.7778,
                8.8889, 10.0000]) freq, norm=False, shift=False
        tensor([0.0000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889,
                1.0000]) freq, norm=True, shift=False
        tensor([-5.0000, -3.8889, -2.7778, -1.6667, -0.5556,  0.5556,  1.6667,  2.7778,
                3.8889,  5.0000]) freq, norm=False, shift=True

    """

    d = 1. / fs

    if n % 2 == 0:
        pp = th.arange(0, n // 2, dtype=dtype, device=device)
        pn = th.arange(-(n // 2), 0, dtype=dtype, device=device)
    else:
        pp = th.arange(0, (n + 1) // 2, dtype=dtype, device=device)
        pn = th.arange(-(n // 2), 0, dtype=dtype, device=device)

    if shift:
        f = th.cat((pn, pp))
    else:
        f = th.cat((pp, pn))

    if norm:
        f = f / n
    else:
        f = f / (n * d)

    return f


def fftshift(x, dim=None):
    r"""Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

    Parameters
    ----------
    x : tensor
        Input tensor.
    dim : int, optional
        Axes over which to shift. (Default is None, which shifts all axes.)

    Returns
    -------
    y : tensor
        The shifted tensor.

    See Also
    --------
    ifftshift : The inverse of `fftshift`.

    Examples
    --------
    ::

        import numpy as np
        import torch as th
        import torchbox as tb

        x = [1, 2, 3, 4, 5, 6]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x)
        print(y)

        x = [1, 2, 3, 4, 5, 6, 7]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x)
        print(y)

        dim = (0, 1)  # dim = 0, dim = 1
        x = [[1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]]
        y = np.fft.fftshift(x, dim)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x, dim)
        print(y)


        x = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7]]
        y = np.fft.fftshift(x, dim)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x, dim)
        print(y)

    """

    if dim is None:
        dim = tuple(range(x.dim()))
    elif type(dim) is int:
        dim = tuple([dim])
    for a in dim:
        n = x.size(a)
        p = int(n / 2.)
        x = th.roll(x, p, dims=a)
    return x


def ifftshift(x, dim=None):
    r"""Shift the zero-frequency component back.

    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.

    Parameters
    ----------
    x : tensor
        The input tensor.
    dim : int, optional
        Axes over which to shift. (Default is None, which shifts all axes.)

    Returns
    -------
    y : tensor
        The shifted tensor.

    See Also
    --------
    fftshift : The inverse of `ifftshift`.

    Examples
    --------
    ::

        import numpy as np
        import torch as th
        import torchbox as tb

        x = [1, 2, 3, 4, 5, 6]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x)
        print(y)

        x = [1, 2, 3, 4, 5, 6, 7]
        y = np.fft.fftshift(x)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x)
        print(y)

        dim = (0, 1)  # dim = 0, dim = 1
        x = [[1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6]]
        y = np.fft.fftshift(x, dim)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x, dim)
        print(y)


        x = [[1, 2, 3, 4, 5, 6, 7], [0, 2, 3, 4, 5, 6, 7]]
        y = np.fft.fftshift(x, dim)
        print(y)
        x = th.tensor(x)
        y = tb.fftshift(x, dim)
        print(y)

    """

    if dim is None:
        dim = tuple(range(x.dim()))
    elif type(dim) is int:
        dim = tuple([dim])
    for a in dim:
        n = x.size(a)
        p = int((n + 1) / 2.)
        x = th.roll(x, p, dims=a)
    return x


def padfft(X, nfft=None, dim=0, shift=False):
    r"""PADFT Pad array for doing FFT or IFFT

    PADFT Pad array for doing FFT or IFFT

    Parameters
    ----------
    X : tensor
        Data to be padded.
    nfft : int or None
        Padding size.
    dim : int, optional
        Padding dimension. (the default is 0)
    shift : bool, optional
        Whether to shift the frequency (the default is False)

    Returns
    -------
    y : tensor
        The padded tensor.
    """

    if dim is None:
        dim = 0

    Nx = X.size(dim)

    if nfft < Nx:
        raise ValueError('Output size is smaller than input size!')

    pad = list(X.size())

    Np = int(np.uint(nfft - Nx))

    if shift:
        pad[dim] = int(np.fix((Np + 1) / 2.))
        Z = th.zeros(pad, dtype=X.dtype, device=X.device)
        X = th.cat((Z, X), dim=dim)
        pad[dim] = Np - pad[dim]
        Z = th.zeros(pad, dtype=X.dtype, device=X.device)
        X = th.cat((X, Z), dim=dim)
    else:
        pad[dim] = Np
        Z = th.zeros(pad, dtype=X.dtype, device=X.device)
        X = th.cat((X, Z), dim=dim)

    return X


def fft(x, n=None, norm="backward", shift=False, **kwargs):
    r"""FFT in torchbox

    FFT in torchbox, both real and complex valued tensors are supported.

    Parameters
    ----------
    x : tensor
        When :attr:`x` is complex, it can be either in real-representation format or complex-representation format.
    n : int, optional
        The number of fft points (the default is None --> equals to signal dimension)
    norm : None or str, optional
        Normalization mode. For the forward transform (fft()), these correspond to:
        "forward" - normalize by ``1/n``; "backward" - no normalization (default); "ortho" - normalize by ``1/sqrt(n)`` (making the FFT orthonormal).
    shift : bool, optional
        shift the zero frequency to center (the default is False)
    cdim : int or None
        If :attr:`x` is complex-valued, :attr:`cdim` is ignored. If :attr:`x` is real-valued and :attr:`cdim` is integer
        then :attr:`x` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex dimension;
        otherwise (None), :attr:`x` will be treated as real-valued.
    dim : int, optional
        axis of fft operation (the default is 0, which means the first dimension)
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`x` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.

    Returns
    -------
    y : tensor
        fft results tensor with the same type as :attr:`x`

    Raises
    ------
    ValueError
        nfft is small than signal dimension.

    see also :func:`ifft`, :func:`fftfreq`, :func:`freq`.

    Examples
    ---------

    .. image:: ./_static/FFTIFFTdemo.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torch as th
        import torchbox as tb
        import matplotlib.pyplot as plt

        shift = True
        frq = [10, 10]
        amp = [0.8, 0.6]
        Fs = 80
        Ts = 2.
        Ns = int(Fs * Ts)

        t = th.linspace(-Ts / 2., Ts / 2., Ns).reshape(Ns, 1)
        f = tb.freq(Ns, Fs, shift=shift)
        f = tb.fftfreq(Ns, Fs, norm=False, shift=shift)

        # ---complex vector in real representation format
        x = amp[0] * th.cos(2. * th.pi * frq[0] * t) + 1j * amp[1] * th.sin(2. * th.pi * frq[1] * t)

        # ---do fft
        Xc = tb.fft(x, n=Ns, cdim=None, dim=0, keepcdim=False, shift=shift)

        # ~~~get real and imaginary part
        xreal = tb.real(x, cdim=None, keepcdim=False)
        ximag = tb.imag(x, cdim=None, keepcdim=False)
        Xreal = tb.real(Xc, cdim=None, keepcdim=False)
        Ximag = tb.imag(Xc, cdim=None, keepcdim=False)

        # ---do ifft
        x̂ = tb.ifft(Xc, n=Ns, cdim=None, dim=0, keepcdim=False, shift=shift)
        
        # ~~~get real and imaginary part
        x̂real = tb.real(x̂, cdim=None, keepcdim=False)
        x̂imag = tb.imag(x̂, cdim=None, keepcdim=False)

        plt.figure()
        plt.subplot(131)
        plt.grid()
        plt.plot(t, xreal)
        plt.plot(t, ximag)
        plt.legend(['real', 'imag'])
        plt.title('signal in time domain')
        plt.subplot(132)
        plt.grid()
        plt.plot(f, Xreal)
        plt.plot(f, Ximag)
        plt.legend(['real', 'imag'])
        plt.title('signal in frequency domain')
        plt.subplot(133)
        plt.grid()
        plt.plot(t, x̂real)
        plt.plot(t, x̂imag)
        plt.legend(['real', 'imag'])
        plt.title('reconstructed signal')
        plt.show()

        # ---complex vector in real representation format
        x = tb.c2r(x, cdim=-1)

        # ---do fft
        Xc = tb.fft(x, n=Ns, cdim=-1, dim=0, keepcdim=False, shift=shift)

        # ~~~get real and imaginary part
        xreal = tb.real(x, cdim=-1, keepcdim=False)
        ximag = tb.imag(x, cdim=-1, keepcdim=False)
        Xreal = tb.real(Xc, cdim=-1, keepcdim=False)
        Ximag = tb.imag(Xc, cdim=-1, keepcdim=False)

        # ---do ifft
        x̂ = tb.ifft(Xc, n=Ns, cdim=-1, dim=0, keepcdim=False, shift=shift)
        
        # ~~~get real and imaginary part
        x̂real = tb.real(x̂, cdim=-1, keepcdim=False)
        x̂imag = tb.imag(x̂, cdim=-1, keepcdim=False)

        plt.figure()
        plt.subplot(131)
        plt.grid()
        plt.plot(t, xreal)
        plt.plot(t, ximag)
        plt.legend(['real', 'imag'])
        plt.title('signal in time domain')
        plt.subplot(132)
        plt.grid()
        plt.plot(f, Xreal)
        plt.plot(f, Ximag)
        plt.legend(['real', 'imag'])
        plt.title('signal in frequency domain')
        plt.subplot(133)
        plt.grid()
        plt.plot(t, x̂real)
        plt.plot(t, x̂imag)
        plt.legend(['real', 'imag'])
        plt.title('reconstructed signal')
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
        dim = 0

    if 'keepcdim' in kwargs:
        keepcdim = kwargs['keepcdim']
    elif 'keepcaxis' in kwargs:
        keepcdim = kwargs['keepcaxis']
    else:
        keepcdim = False

    if norm is None:
        norm = 'backward'

    CplxRealflag = False
    if th.is_complex(x):  # complex in complex
        pass
    else:
        if cdim is None:  # real
            pass
        else:  # complex in real
            CplxRealflag = True
            x = tb.r2c(x, cdim=cdim, keepcdim=keepcdim)

    d = x.size(dim)
    if n is None:
        n = d
    if d < n:
        x = padfft(x, n, dim, shift)
    elif d > n:
        raise ValueError('nfft is small than signal dimension!')

    if shift:
        y = thfft.fftshift(thfft.fft(thfft.fftshift(x, dim=dim), n=n, dim=dim, norm=norm), dim=dim)
    else:
        y = thfft.fft(x, n=n, dim=dim, norm=norm)

    if CplxRealflag:
        y = tb.c2r(y, cdim=cdim, keepcdim=not keepcdim)

    return y


def ifft(x, n=None, norm="backward", shift=False, **kwargs):
    r"""IFFT in torchbox

    IFFT in torchbox, both real and complex valued tensors are supported.

    Parameters
    ----------
    x : tensor
        When :attr:`x` is complex, it can be either in real-representation format or complex-representation format.
    n : int, optional
        The number of ifft points (the default is None --> equals to signal dimension)
    norm : bool, optional
        Normalization mode. For the backward transform (ifft()), these correspond to: "forward" - no normalization;
         "backward" - normalize by ``1/n`` (default); "ortho" - normalize by 1``/sqrt(n)`` (making the IFFT orthonormal).
    shift : bool, optional
        shift the zero frequency to center (the default is False)
    cdim : int or None
        If :attr:`x` is complex-valued, :attr:`cdim` is ignored. If :attr:`x` is real-valued and :attr:`cdim` is integer
        then :attr:`x` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis;
        otherwise (None), :attr:`x` will be treated as real-valued.
    dim : int, optional
        axis of fft operation (the default is 0, which means the first dimension)
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`x` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.

    Returns
    -------
    y : tensor
        ifft results tensor with the same type as :attr:`x`

    Raises
    ------
    ValueError
        nfft is small than signal dimension.

    see also :func:`fft`, :func:`fftfreq`, :func:`freq`. see :func:`fft` for examples. 

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
        
    if 'keepcdim' in kwargs:
        keepcdim = kwargs['keepcdim']
    elif 'keepcaxis' in kwargs:
        keepcdim = kwargs['keepcaxis']
    else:
        keepcdim = False

    if norm is None:
        norm = 'backward'

    CplxRealflag = False
    if th.is_complex(x):  # complex in complex
        pass
    else:
        if cdim is None:  # real
            pass
        else:  # complex in real
            CplxRealflag = True
            x = tb.r2c(x, cdim=cdim, keepcdim=keepcdim)

    if shift:
        y = thfft.ifftshift(thfft.ifft(thfft.ifftshift(x, dim=dim), n=n, dim=dim, norm=norm), dim=dim)
    else:
        y = thfft.ifft(x, n=n, dim=dim, norm=norm)

    if CplxRealflag:
        y = tb.c2r(y, cdim=cdim, keepcdim=not keepcdim)

    return y


if __name__ == '__main__':

    print(th.__version__)
    nfft = 4
    ftshift = False
    x1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    print(x1.shape)
    y1 = np.fft.fft(x1, n=nfft, axis=1, norm=None)
    print(y1, y1.shape)
    x1 = np.fft.ifft(y1, n=nfft, axis=1, norm=None)
    print(x1)

    x2 = th.tensor(x1, dtype=th.float32)
    x2 = th.stack([x2, th.zeros(x2.size())], dim=-1)

    y2 = fft(x2, n=nfft, dim=1, norm=None, shift=ftshift)
    print(y2, y2.shape)
    x2 = ifft(y2, n=nfft, dim=1, norm=None, shift=ftshift)
    print(x2)
