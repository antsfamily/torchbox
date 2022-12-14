#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : complex_functions.py
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
from torch.nn.functional import relu, leaky_relu, max_pool2d, max_pool1d, dropout, dropout2d, upsample


def complex_relu(input, inplace=False):
    return th.stack((relu(input[..., 0], inplace),
                     relu(input[..., 1], inplace)), dim=-1)


def complex_leaky_relu(input, negative_slope=(0.01, 0.01), inplace=False):
    return th.stack((leaky_relu(input[..., 0], negative_slope[0], inplace),
                     leaky_relu(input[..., 1], negative_slope[1], inplace)), dim=-1)


def complex_max_pool2d(input, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):

    return th.stack((max_pool2d(input[..., 0], kernel_size, stride, padding, dilation, ceil_mode, return_indices),
                     max_pool2d(input[..., 1], kernel_size, stride, padding, dilation, ceil_mode, return_indices)), dim=-1)


def complex_max_pool1d(input, kernel_size, stride=None, padding=0,
                       dilation=1, ceil_mode=False, return_indices=False):

    return th.stack((max_pool1d(input[..., 0], kernel_size, stride, padding, dilation, ceil_mode, return_indices),
                     max_pool1d(input[..., 1], kernel_size, stride, padding, dilation, ceil_mode, return_indices)), dim=-1)


def complex_dropout(input, p=0.5, training=True, inplace=False):
    return th.stack((dropout(input[..., 0], p, training, inplace),
                     dropout(input[..., 1], p, training, inplace)), dim=-1)


def complex_dropout2d(input, p=0.5, training=True, inplace=False):
    return th.stack((dropout2d(input[..., 0], p, training, inplace),
                     dropout2d(input[..., 1], p, training, inplace)), dim=-1)


def complex_upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    return th.stack((upsample(input[..., 0], size, scale_factor, mode, align_corners),
                     upsample(input[..., 1], size, scale_factor, mode, align_corners)), dim=-1)

