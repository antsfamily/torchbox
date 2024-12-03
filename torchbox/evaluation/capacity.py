#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : capacity.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Mon Sep 04 2023
# @version   : 0.0
# @license   : GNU General Public License (GPL)
# @note      : 
# 
# The GNU General Public License (GPL)
# Copyright (C) 2013- Zhi Liu
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program. If not, see <https://www.gnu.org/licenses/>.
#


def capacity(H, snr, cdim, dim=(-2, -1), keepdim=False, reduction='mean'):
    """computes capacity of channel

    MIMO-OFDM Wireless Communications with MATLAB

    Parameters
    ----------
    H : Tensor
        the input channel
    snr : float
        the signal-to-noise ratio
    cdim : int or None
        If :attr:`H` is complex-valued, :attr:`cdim` is ignored. 
        If :attr:`H` is real-valued and :attr:`cdim` is an integer
        then :attr:`H` will be treated as complex-valued, in this case, :attr:`cdim` specifies the complex axis.
    dim : int or None
        The dimension indexes of antenna of BS and MS. 
        The default is ``(-2, -1)``. 
    keepdim : bool
        keep dimensions? (include complex dim, defalut is :obj:`False`)
    reduction : str or None, optional
        The operation mode of reduction, :obj:`None`, ``'mean'`` or ``'sum'`` (the default is 'mean')
        
    """
    raise ValueError('Not support yet!')


