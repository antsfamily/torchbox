#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : fourier.py
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
import torchbox as tb


class FourierLoss(th.nn.Module):
    r"""Fourier Domain Loss

    Compute loss in Fourier domain. Given input :math:`{\bm P}`, target :math:`\bm G`, 
    
    .. math::
       L = g({\mathcal F}({\bm P}), {\mathcal F}({\bm G}))
    
    where, :math:`{\bm P}`, :math:`\bm G` can be real-valued and complex-valued data, :math:`g(\cdot)` is a
    function, such as mean square error, absolute error, ...

    Parameters
    ----------
    cdim : int, optional
        If data is complex-valued but represented as real tensors, 
        you should specify the dimension. Otherwise, set it to None, defaults is None.
        For example, :math:`{\bm X}_c\in {\mathbb C}^{N\times C\times H\times W}` is
        represented as a real-valued tensor :math:`{\bm X}_r\in {\mathbb R}^{N\times C\times H\times W\ times 2}`,
        then :attr:`cdim` equals to -1 or 4.
    ftdim : tuple, None, optional
        the dimensions for Fourier transformation. by default (-2, -1).
    iftdim : tuple, None, optional
        the dimension for inverse Fourier transformation, by default None.
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    ftn : int, None, optional
        the number of points for Fourier transformation, by default None
    iftn : int, None, optional
        the number of points for inverse Fourier transformation, by default None
    ftnorm : str, None, optional
        the normalization method for Fourier transformation, by default None
        - "forward" - normalize by 1/n
        - "backward" - no normalization
        - "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
    iftnorm : str, None, optional
        the normalization method for inverse Fourier transformation, by default None
        - "forward" - no normalization
        - "backward" - normalize by 1/n
        - "ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)
    err : str, loss function, optional
        ``'MSE'``, ``'MAE'`` or torch's loss function, by default ``'mse'``
    reduction : str, optional
        reduction behavior, ``'sum'`` or ``'mean'``, by default ``'mean'``

    please see :func:`th.nn.fft.fft` and :func:`th.nn.fft.ifft`.

    Examples
    ---------

    Compute loss of data in real and complex representation, respectively.

    ::

        th.manual_seed(2020)
        xr = th.randn(10, 2, 4, 4) * 10000
        yr = th.randn(10, 2, 4, 4) * 10000
        xc = xr[:, [0], ...] + 1j * xr[:, [1], ...]
        yc = yr[:, [0], ...] + 1j * yr[:, [1], ...]

        flossr = FourierLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
        flossc = FourierLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        flossr = FourierLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
        flossc = FourierLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        # ---output
        tensor(7.2681e+08)
        tensor(7.2681e+08)
        tensor(45425624.)
        tensor(45425624.)

    """

    def __init__(self, err='mse', cdim=None, ftdim=(-2, -1), iftdim=None, keepcdim=False, ftn=None, iftn=None, ftnorm=None, iftnorm=None, reduction='mean'):
        super(FourierLoss, self).__init__()
        self.cdim = cdim
        self.ftdim = [ftdim] if (type(ftdim) is not list and type(ftdim) is not tuple) else ftdim
        self.iftdim = [iftdim] if (type(iftdim) is not list and type(iftdim) is not tuple) else iftdim
        self.ftn = [ftn] if (type(ftn) is not list and type(ftn) is not tuple) else ftn
        self.iftn = [iftn] if (type(iftn) is not list and type(iftn) is not tuple) else iftn
        self.ftnorm = [ftnorm] if (type(ftnorm) is not list and type(ftnorm) is not tuple) else ftnorm
        self.iftnorm = [iftnorm] if (type(iftnorm) is not list and type(iftnorm) is not tuple) else iftnorm
        self.keepcdim = keepcdim
        self.reduction = reduction

        if err in ['sse', 'SSE', 'Sse']:
            self.err = tb.SSELoss(reduction=self.reduction)
        if err in ['mse', 'MSE', 'Mse']:
            self.err = tb.MSELoss(reduction=self.reduction)
        if err in ['sae', 'SAE', 'Sae']:
            self.err = tb.SAELoss(reduction=self.reduction)
        if err in ['mae', 'MAE', 'Mae']:
            self.err = tb.MAELoss(reduction=self.reduction)
        if str(type(err)).find('torch.nn.modules.loss') > 0:
            self.err = err

    def forward(self, P, G):
        if self.cdim is not None:
            P = tb.r2c(P, cdim=self.cdim, keepcdim=self.keepcdim)
            G = tb.r2c(G, cdim=self.cdim, keepcdim=self.keepcdim)

        for dim, n, norm in zip(self.ftdim, self.ftn, self.ftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.fft(P, n=n, dim=dim, norm=norm)
                G = th.fft.fft(G, n=n, dim=dim, norm=norm)

        for dim, n, norm in zip(self.iftdim, self.iftn, self.iftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.ifft(P, n=n, dim=dim, norm=norm)
                G = th.fft.ifft(G, n=n, dim=dim, norm=norm)

        P = tb.c2r(P, cdim=-1)
        G = tb.c2r(G, cdim=-1)

        return self.err(P, G)


class FourierAmplitudeLoss(th.nn.Module):
    r"""Fourier Domain Amplitude Loss

    compute amplitude loss in fourier domain.

    Parameters
    ----------
    err : str, loss function, optional
        ``'MSE'``, ``'MAE'`` or torch's loss function, by default ``'mse'``.
    cdim : int, optional
        If data is complex-valued but represented as real tensors, 
        you should specify the dimension. Otherwise, set it to None, defaults is None.
        For example, :math:`{\bm X}_c\in {\mathbb C}^{N\times C\times H\times W}` is
        represented as a real-valued tensor :math:`{\bm X}_r\in {\mathbb R}^{N\times C\times H\times W\ times 2}`,
        then :attr:`cdim` equals to -1 or 4.
    ftdim : tuple, None, optional
        the dimensions for Fourier transformation. by default (-2, -1).
    iftdim : tuple, None, optional
        the dimension for inverse Fourier transformation, by default None.
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    ftn : int, None, optional
        the number of points for Fourier transformation, by default None
    iftn : int, None, optional
        the number of points for inverse Fourier transformation, by default None
    ftnorm : str, None, optional
        the normalization method for Fourier transformation, by default None
        - "forward" - normalize by 1/n
        - "backward" - no normalization
        - "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
    iftnorm : str, None, optional
        the normalization method for inverse Fourier transformation, by default None
        - "forward" - no normalization
        - "backward" - normalize by 1/n
        - "ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)
    reduction : str, optional
        reduction behavior, ``'sum'`` or ``'mean'``, by default ``'mean'``

    please see :func:`th.nn.fft.fft` and :func:`th.nn.fft.ifft`.

    Examples
    ---------

    Compute loss of data in real and complex representation, respectively.

    ::

        th.manual_seed(2020)
        xr = th.randn(10, 2, 4, 4) * 10000
        yr = th.randn(10, 2, 4, 4) * 10000
        xc = xr[:, [0], ...] + 1j * xr[:, [1], ...]
        yc = yr[:, [0], ...] + 1j * yr[:, [1], ...]

        flossr = FourierAmplitudeLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
        flossc = FourierAmplitudeLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        flossr = FourierAmplitudeLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
        flossc = FourierAmplitudeLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        # ---output
        tensor(2.8548e+08)
        tensor(2.8548e+08)
        tensor(17842250.)
        tensor(17842250.)

    """

    def __init__(self, err='mse', cdim=None, ftdim=(-2, -1), iftdim=None, keepcdim=False, ftn=None, iftn=None, ftnorm=None, iftnorm=None, reduction='mean'):
        super(FourierAmplitudeLoss, self).__init__()
        self.cdim = cdim
        self.ftdim = [ftdim] if (type(ftdim) is not list and type(ftdim) is not tuple) else ftdim
        self.iftdim = [iftdim] if (type(iftdim) is not list and type(iftdim) is not tuple) else iftdim
        self.ftn = [ftn] if (type(ftn) is not list and type(ftn) is not tuple) else ftn
        self.iftn = [iftn] if (type(iftn) is not list and type(iftn) is not tuple) else iftn
        self.ftnorm = [ftnorm] if (type(ftnorm) is not list and type(ftnorm) is not tuple) else ftnorm
        self.iftnorm = [iftnorm] if (type(iftnorm) is not list and type(iftnorm) is not tuple) else iftnorm
        self.keepcdim = keepcdim
        self.reduction = reduction

        if err in ['sse', 'SSE', 'Sse']:
            self.err = tb.SSELoss(reduction=self.reduction)
        if err in ['mse', 'MSE', 'Mse']:
            self.err = tb.MSELoss(reduction=self.reduction)
        if err in ['sae', 'SAE', 'Sae']:
            self.err = tb.SAELoss(reduction=self.reduction)
        if err in ['mae', 'MAE', 'Mae']:
            self.err = tb.MAELoss(reduction=self.reduction)
        if str(type(err)).find('torch.nn.modules.loss') > 0:
            self.err = err

    def forward(self, P, G):
        if self.cdim is not None:
            P = tb.r2c(P, cdim=self.cdim, keepcdim=self.keepcdim)
            G = tb.r2c(G, cdim=self.cdim, keepcdim=self.keepcdim)

        for dim, n, norm in zip(self.ftdim, self.ftn, self.ftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.fft(P, n=n, dim=dim, norm=norm)
                G = th.fft.fft(G, n=n, dim=dim, norm=norm)

        for dim, n, norm in zip(self.iftdim, self.iftn, self.iftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.ifft(P, n=n, dim=dim, norm=norm)
                G = th.fft.ifft(G, n=n, dim=dim, norm=norm)

        P, G = P.abs(), G.abs()

        return self.err(P, G)


class FourierPhaseLoss(th.nn.Module):
    r"""Fourier Domain Phase Loss

    compute phase loss in fourier domain.

    Parameters
    ----------
    err : str, loss function, optional
        ``'MSE'``, ``'MAE'`` or torch's loss function, by default ``'mse'``
    cdim : int, optional
        If data is complex-valued but represented as real tensors, 
        you should specify the dimension. Otherwise, set it to None, defaults is None.
        For example, :math:`{\bm X}_c\in {\mathbb C}^{N\times C\times H\times W}` is
        represented as a real-valued tensor :math:`{\bm X}_r\in {\mathbb R}^{N\times C\times H\times W\ times 2}`,
        then :attr:`cdim` equals to -1 or 4.
    ftdim : tuple, None, optional
        the dimensions for Fourier transformation. by default (-2, -1).
    iftdim : tuple, None, optional
        the dimension for inverse Fourier transformation, by default None.
    keepcdim : bool
        If :obj:`True`, the complex dimension will be keeped. Only works when :attr:`X` is complex-valued tensor 
        but represents in real format. Default is :obj:`False`.
    ftn : int, None, optional
        the number of points for Fourier transformation, by default None
    iftn : int, None, optional
        the number of points for inverse Fourier transformation, by default None
    ftnorm : str, None, optional
        the normalization method for Fourier transformation, by default None
        - "forward" - normalize by 1/n
        - "backward" - no normalization
        - "ortho" - normalize by 1/sqrt(n) (making the FFT orthonormal)
    iftnorm : str, None, optional
        the normalization method for inverse Fourier transformation, by default None
        - "forward" - no normalization
        - "backward" - normalize by 1/n
        - "ortho" - normalize by 1/sqrt(n) (making the IFFT orthonormal)
    reduction : str, optional
        reduction behavior, ``'sum'`` or ``'mean'``, by default ``'mean'``

    please see :func:`th.nn.fft.fft` and :func:`th.nn.fft.ifft`.

    Examples
    ---------

    Compute loss of data in real and complex representation, respectively.

    ::

        th.manual_seed(2020)
        xr = th.randn(10, 2, 4, 4) * 10000
        yr = th.randn(10, 2, 4, 4) * 10000
        xc = xr[:, [0], ...] + 1j * xr[:, [1], ...]
        yc = yr[:, [0], ...] + 1j * yr[:, [1], ...]

        flossr = FourierPhaseLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
        flossc = FourierPhaseLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        flossr = FourierPhaseLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
        flossc = FourierPhaseLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
        print(flossr(xr, yr))
        print(flossc(xc, yc))

        # ---output
        tensor(6.6797)
        tensor(6.6797)
        tensor(6.6797)
        tensor(6.6797)

    """

    def __init__(self, err='mse', cdim=None, ftdim=(-2, -1), iftdim=None, keepcdim=False, ftn=None, iftn=None, ftnorm=None, iftnorm=None, reduction='mean'):
        super(FourierPhaseLoss, self).__init__()
        self.cdim = cdim
        self.ftdim = [ftdim] if (type(ftdim) is not list and type(ftdim) is not tuple) else ftdim
        self.iftdim = [iftdim] if (type(iftdim) is not list and type(iftdim) is not tuple) else iftdim
        self.ftn = [ftn] if (type(ftn) is not list and type(ftn) is not tuple) else ftn
        self.iftn = [iftn] if (type(iftn) is not list and type(iftn) is not tuple) else iftn
        self.ftnorm = [ftnorm] if (type(ftnorm) is not list and type(ftnorm) is not tuple) else ftnorm
        self.iftnorm = [iftnorm] if (type(iftnorm) is not list and type(iftnorm) is not tuple) else iftnorm
        self.keepcdim = keepcdim
        self.reduction = reduction

        if err in ['sse', 'SSE', 'Sse']:
            self.err = tb.SSELoss(reduction=self.reduction)
        if err in ['mse', 'MSE', 'Mse']:
            self.err = tb.MSELoss(reduction=self.reduction)
        if err in ['sae', 'SAE', 'Sae']:
            self.err = tb.SAELoss(reduction=self.reduction)
        if err in ['mae', 'MAE', 'Mae']:
            self.err = tb.MAELoss(reduction=self.reduction)
        if str(type(err)).find('torch.nn.modules.loss') > 0:
            self.err = err

    def forward(self, P, G):
        if self.cdim is not None:
            P = tb.r2c(P, cdim=self.cdim, keepcdim=self.keepcdim)
            G = tb.r2c(G, cdim=self.cdim, keepcdim=self.keepcdim)

        for dim, n, norm in zip(self.ftdim, self.ftn, self.ftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.fft(P, n=n, dim=dim, norm=norm)
                G = th.fft.fft(G, n=n, dim=dim, norm=norm)

        for dim, n, norm in zip(self.iftdim, self.iftn, self.iftnorm):
            if dim is None:
                pass
            else:
                P = th.fft.ifft(P, n=n, dim=dim, norm=norm)
                G = th.fft.ifft(G, n=n, dim=dim, norm=norm)

        P, G = P.angle(), G.angle()

        return self.err(P, G)


class FourierNormLoss(th.nn.Module):
    r"""FourierNormLoss

    .. math::
       C = \frac{{\rm E}(|I|^2)}{[E(|I|)]^2}

    see Fast Fourier domain optimization using hybrid

    """

    def __init__(self, reduction='mean', p=1.5):
        super(FourierNormLoss, self).__init__()
        self.reduction = reduction
        self.p = p

    def forward(self, X, w=None):
        r"""[summary]

        [description]

        Parameters
        ----------
        X : Tensor
            After fft in azimuth
        w : Tensor, optional
            weight

        Returns
        -------
        float
            loss
        """

        if th.is_complex(X):
            X = X.abs()
        elif X.shape[-1] == 2:
            X = th.view_as_complex(X)
            X = X.abs()

        if w is None:
            wshape = [1] * (X.dim())
            wshape[-2] = X.size(-2)
            w = th.ones(wshape, device=X.device, dtype=X.dtype)
        fv = th.sum((th.sum(w * X, dim=-2)).pow(self.p), dim=-1)

        if self.reduction == 'mean':
            C = th.mean(fv)
        if self.reduction == 'sum':
            C = th.sum(fv)
        return C


if __name__ == '__main__':

    th.manual_seed(2020)
    xr = th.randn(10, 2, 4, 4) * 10000
    yr = th.randn(10, 2, 4, 4) * 10000
    xc = xr[:, [0], ...] + 1j * xr[:, [1], ...]
    yc = yr[:, [0], ...] + 1j * yr[:, [1], ...]

    flossr = FourierLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
    flossc = FourierLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
    print(flossr(xr, yr))
    print(flossc(xc, yc))

    flossr = FourierLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
    flossc = FourierLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
    print(flossr(xr, yr))
    print(flossc(xc, yc))


    flossr = FourierAmplitudeLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
    flossc = FourierAmplitudeLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
    print(flossr(xr, yr))
    print(flossc(xc, yc))

    flossr = FourierAmplitudeLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
    flossc = FourierAmplitudeLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
    print(flossr(xr, yr))
    print(flossc(xc, yc))


    flossr = FourierPhaseLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
    flossc = FourierPhaseLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm=None, iftnorm=None, err='mse', reduction='mean')
    print(flossr(xr, yr))
    print(flossc(xc, yc))

    flossr = FourierPhaseLoss(cdim=1, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
    flossc = FourierPhaseLoss(cdim=None, ftdim=(-2, -1), iftdim=None, ftn=None, iftn=None, ftnorm='forward', iftnorm=None, err='mse', reduction='mean')
    print(flossr(xr, yr))
    print(flossc(xc, yc))
