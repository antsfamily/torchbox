#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : balanceconv2d.py
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

import numpy  as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd

from torch.distributions.bernoulli import Bernoulli


class BalaConv2d(_ConvNd):
    r"""Applies a 2D Balanced convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        {\bm Z}_{n_o, c_i, h_o, w_o} = \sum_{h=0}^{H_k-1}\sum_{w=0}^{W_k-1} \left[{\bm I}_{n_o, c_i, h_o + h - 1, w_o + w - 1} + {\bm K}_{c_o, h, w}
           - {\bm I}_{n_o, c_i, h_o + h - 1, w_o + w - 1} \cdot {\bm K}_{c_o, h, w}\right].
       :label: equ-BalancedConv2d


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=False, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BalaConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.weight_ones = torch.ones_like(self.weight)
        if self.bias is not None:
            self.bias_zeros = torch.zeros_like(self.bias)
            self.bias_zeros.requires_grad = False
        else:
            self.bias_zeros = None
        self.weight_ones.requires_grad = False
        self.alpha = torch.ones_like(self.weight)
        self.alpha.requires_grad = True
        self.m = torch.nn.Tanh()


    def forward(self, input):
        input_ones = torch.ones_like(input)

        if self.weight_ones.device != self.weight.device:
            self.weight_ones = self.weight_ones.to(self.weight.device)
        if self.alpha.device != self.weight.device:
            self.alpha = self.alpha.to(self.weight.device)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            conv2d = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                              self.weight, self.bias, self.stride,
                              _pair(0), self.dilation, self.groups)
            term1 = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                             self.weight_ones, self.bias_zeros, self.stride,
                             _pair(0), self.dilation, self.groups)
            term2 = F.conv2d(F.pad(input_ones, expanded_padding, mode='circular'),
                             self.weights, self.bias_zeros, self.stride,
                             _pair(0), self.dilation, self.groups)
        else:
            w = self.m(self.weight)

            conv2d = F.conv2d(input, w, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
            # term1 = F.conv2d(input, self.weight_ones, self.bias_zeros, self.stride,
            #                  self.padding, self.dilation, self.groups)
            # term2 = F.conv2d(input_ones, self.weight, self.bias_zeros, self.stride,
            #                  self.padding, self.dilation, self.groups)

        # conv2d = term1 - conv2d
        conv2d = conv2d

        return conv2d


if __name__ == '__main__':

    import torch.nn as nn
    import torchbox as tb

    # With square kernels and equal stride
    m = nn.Conv2d(16, 32, 3, stride=2)

    # non-square kernels and unequal stride and with padding
    m = nn.Conv2d(16, 32, (3, 5), stride=(2, 1), padding=(4, 2))

    # non-square kernels and unequal stride and with padding and dilation
    m = nn.Conv2d(16, 32, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))

    input = torch.randn(20, 16, 50, 100)
    output1 = m(input)

    print(input.size(), output1.size())

    # -----------------------------------------------------------------------

    # With square kernels and equal stride
    m = tb.nn.BalaConv2d(16, 32, 3, stride=2)

    # non-square kernels and unequal stride and with padding
    m = tb.nn.BalaConv2d(16, 32, (3, 5), stride=(2, 1), padding=(4, 2))

    # non-square kernels and unequal stride and with padding and dilation
    m = tb.nn.BalaConv2d(16, 32, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))

    input = torch.randn(20, 16, 50, 100)
    output2 = m(input)

    print(input.size(), output2.size())

    print(output2)
