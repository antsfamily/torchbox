#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : entropy.py
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


class ChnlCapCor(th.nn.Module):
    r"""computes the capacity and correlation metric value of channel

    see "MIMO-OFDM Wireless Communications with MATLAB (Yong Soo Cho, Jaekwon Kim, Won Young Yang etc.)",
    

    Parameters
    ----------
    EsN0 : float
        the ratio of symbol energy to noise power spectral density, :math:`E_s/N_0({\rm dB}) = E_b/N_0 + 10{\rm log}_{10}K`
        :math:`E_s/N_0({\rm dB})=10{\rm log}_{10}(T_{\rm symbol}/T_{\rm sample}) + {\rm SNR}(dB)`, default is 30
    rank : int
        the rank of channel ( `what is <https://www.salimwireless.com/2022/11/channel-matrix-in-communication.html>`_ ), by default 4
    way : int
        computation mode: ``'det'``, ``'hadineq'`` (Hadamard inequality), ``'inv'`` (default)
    cdim : int or None
        If :attr:`H` is complex-valued, :attr:`cdim` is ignored. 
        If :attr:`H` is real-valued and :attr:`cdim` is an integer
        then :attr:`H` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis.
    dim : int or None
        The dimension indexes of (sub-carrirer, BS antenna, UE antenna), The default is ``(-3, -2, -1)``. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is 'mean')
    
    Examples
    --------

    Here are demo codes.
    
    ::

        import torch as th
        import torchbox as tb

        th.manual_seed(2020)
        Nt, Nsc, Nbs, Nms = 10, 360, 64, 4
        # generates the ground-truth
        Hg = th.randn(Nt, 2, Nsc, Nbs, Nms)
        # noised version as the predicted
        Hp = tb.awgns(Hg, snrv=10, cdim=1, dim=(-3, -2, -1))

        # complex in real format
        metric = tb.ChnlCapCor(rank=4, cdim=1, dim=(-3, -2, -1), reduction='mean')
        metric.updategt(Hg)
        print(metric.forward(Hp))

        Hg = Hg[:, 0, ...] + 1j * Hg[:, 1, ...]
        Hp = Hp[:, 0, ...] + 1j * Hp[:, 1, ...]
        # complex in complex format
        metric = tb.ChnlCapCor(rank=4, cdim=None, dim=(-3, -2, -1), reduction='mean')
        metric.updategt(Hg)
        print(metric.forward(Hp))
        print(metric.forward(Hg))

        # complex in complex format
        metric = tb.ChnlCapCor(30, rank=4, cdim=None, dim=(-3, -2, -1), reduction=None)
        metric.updategt(Hg)
        capv, corv = metric.forward(Hp)
        print(capv.shape, corv.shape)

        # ---output
        (tensor(21.0226), tensor(0.8575))
        (tensor(21.0226), tensor(0.8575))
        (tensor(21.5848), tensor(1.))
        torch.Size([10]) torch.Size([10, 4])

    """

    def __init__(self, EsN0=30, rank=4, way='inv', cdim=None, dim=None, keepdim=False, reduction='mean'):
        super(ChnlCapCor, self).__init__()
        self.Hg = None
        self.Ug = None
        self.EsN0 = EsN0
        self.rank = rank
        self.way = way
        self.cdim = cdim
        self.dim = dim
        self.keepdim = keepdim
        self.reduction = reduction

    def updategt(self, Hg):
        """update the ground-truth
        
        Parameters
        ----------
        Hg : Tensor
            the ground-truth channel

        """

        if self.cdim is not None:
            dim = tb.rmcdim(Hg.ndim, cdim=self.cdim, dim=self.dim, keepdim=self.keepdim)
            self.Hg = tb.r2c(Hg, cdim=self.cdim, keepdim=self.keepdim)
        else:
            dim = self.dim
            self.Hg = Hg
        scdim, bsdim, uedim = dim
        Ug, _ = tb.pcat(self.Hg, fdim=bsdim, sdim=uedim, isnorm=True, eigbkd='svd')
        self.Ug = Ug

        self.power = ((self.Hg * self.Hg.conj()).real).mean().sqrt()
        self.sigma = self.power / self.EsN0
    
    def forward(self, Hp):
        """forward process

        Parameters
        ----------
        Hp : Tensor
            the predicted/estimated channel.

        Returns
        -------
        capv : scalar or Tensor
            The capacity of the channel.
        corv : scalar or Tensor
            The correlation of the channel.
        """

        if (self.Ug is None) or (self.Hg is None):
            raise ValueError('Please run update first!')

        if self.cdim is not None:
            dim = tb.rmcdim(Hp.ndim, cdim=self.cdim, dim=self.dim, keepdim=self.keepdim)
            Hp = tb.r2c(Hp, cdim=self.cdim, keepdim=self.keepdim)
        else:
            dim = self.dim
        
        scdim, bsdim, uedim = dim
        Nabs, Naue = Hp.shape[bsdim], Hp.shape[uedim]
        Up, _ = tb.pcat(Hp, fdim=bsdim, sdim=uedim, isnorm=True, eigbkd='svd')

        Hg = tb.permute(self.Hg, dim=(bsdim, uedim), mode='matmul', dir='f')

        Hu = Hg.transpose(-2, -1).conj() @ Up[..., :self.rank]
        A = Hu.conj().transpose(-2, -1) @ Hu / self.sigma / Nabs + th.eye(self.rank)

        if self.way.lower() in ['det', 'determinant']:
            capv = th.log2(th.det(A).real).unsqueeze(-1).unsqueeze(-1)
        elif self.way.lower() in ['hadineq', 'hadamard inequality']:
            capv = th.log2(th.prod(th.diagonal(A, dim1=-2, dim2=-1).real, dim=-1, keepdim=True)).unsqueeze(-1)
        elif self.way.lower() in ['inv', 'inverse']:
            capv = th.log2(th.prod(th.diagonal(1. / th.linalg.inv(A), dim1=-2, dim2=-1).real, dim=-1, keepdim=True)).unsqueeze(-1)
        else:
            raise ValueError('Not support way: %s!' % self.way)

        capv = th.mean(capv, dim=scdim, keepdim=True)
        corv = tb.cossim(self.Ug[..., :self.rank], Up[..., :self.rank], mode='abs', cdim=None, dim=-2, keepdim=True, reduction=None)
        corv = th.mean(corv, dim=scdim, keepdim=True)

        rdim = tb.dimreduce(capv.ndim, cdim=None, dim=dim, keepcdim=self.keepdim, reduction=self.reduction)

        capv = tb.reduce(capv, dim=rdim, keepdim=self.keepdim, reduction=self.reduction)
        corv = tb.reduce(corv, dim=rdim, keepdim=self.keepdim, reduction=self.reduction)

        return capv, corv


if __name__ == '__main__':

    import torch as th
    import torchbox as tb

    th.manual_seed(2020)
    Nt, Nsc, Nbs, Nms = 10, 360, 64, 4
    # generates the ground-truth
    Hg = th.randn(Nt, 2, Nsc, Nbs, Nms)
    # noised version as the predicted
    Hp = tb.awgns(Hg, snrv=10, cdim=1, dim=(-3, -2, -1))

    # complex in real format
    metric = tb.ChnlCapCor(rank=4, cdim=1, dim=(-3, -2, -1), reduction='mean')
    metric.updategt(Hg)
    print(metric.forward(Hp))

    Hg = Hg[:, 0, ...] + 1j * Hg[:, 1, ...]
    Hp = Hp[:, 0, ...] + 1j * Hp[:, 1, ...]
    # complex in complex format
    metric = tb.ChnlCapCor(rank=4, cdim=None, dim=(-3, -2, -1), reduction='mean')
    metric.updategt(Hg)
    print(metric.forward(Hp))
    print(metric.forward(Hg))

    # complex in complex format
    metric = tb.ChnlCapCor(30, rank=4, cdim=None, dim=(-3, -2, -1), reduction=None)
    metric.updategt(Hg)
    capv, corv = metric.forward(Hp)
    print(capv.shape, corv.shape)

