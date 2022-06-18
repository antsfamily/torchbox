#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-06 21:14:04
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th


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
        :attr:`cdim` or :attr:`caxis` should be specified. If not, it's set to :obj:`None`, 
        which means :attr:`x` and :attr:`n` are real-valued or complex-valued in complex format.
    keepcdim : int or None, optional
        keep the complex dimension?
    dim : int or None, optional
        Specifies the dimensions for computing SNR, if not specified, it's set to :obj:`None`, 
        which means all the dimensions.
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
        snrv = snr(x, n, cdim=1, dim=(2, 3), keepcdim=True)
        print(snrv)
        snrv = snr(x, n, cdim=1, dim=(2, 3), keepcdim=True, reduction='mean')
        print(snrv)
        x = tb.r2c(x, cdim=1)
        n = tb.r2c(n, cdim=1)
        snrv = snr(x, n, cdim=None, dim=(1, 2), reduction='mean')
        print(snrv)

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

    if 'keepcdim' in kwargs:
        keepcdim = kwargs['keepcdim']
    elif 'keepcaxis' in kwargs:
        keepcdim = kwargs['keepcaxis']
    else:
         keepcdim = False

    if 'reduction' in kwargs:
        reduction = kwargs['reduction']
    else:
        reduction = None
        
    dim = tuple(range(x.dim())) if dim is None else dim

    if th.is_complex(x):  # complex in complex
        Px = th.sum((x * x.conj()).real, dim=dim)
        Pn = th.sum((n * n.conj()).real, dim=dim)
    elif cdim is None:  # real
        Px = th.sum(x**2, dim=dim)
        Pn = th.sum(n**2, dim=dim)
    else: # complex in real
        Px = th.sum(x**2, dim=cdim, keepdim=keepcdim)
        Pn = th.sum(n**2, dim=cdim, keepdim=keepcdim)
        Px = th.sum(Px, dim=dim)
        Pn = th.sum(Pn, dim=dim)
    
    S = 10 * th.log10(Px / Pn)
    if reduction in ['sum', 'SUM']:
        return th.sum(S)
    if reduction in ['mean', 'MEAN']:
        return th.mean(S)
    return S


if __name__ == '__main__':

    import torchbox as tb

    tb.setseed(seed=2020, target='torch')
    x = 10 * th.randn(5, 2, 3, 4)
    n = th.randn(5, 2, 3, 4)
    snrv = snr(x, n, cdim=1, dim=(2, 3), keepcdim=True)
    print(snrv)
    snrv = snr(x, n, cdim=1, dim=(2, 3), keepcdim=True, reduction='mean')
    print(snrv)
    x = tb.r2c(x, cdim=1)
    n = tb.r2c(n, cdim=1)
    snrv = snr(x, n, cdim=None, dim=(1, 2), reduction='mean')
    print(snrv)

