#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : conv_lstms.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Feb 19 2023
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


class ConvLSTMCell(th.nn.Module):
    r"""Cell class for the ConvLSTM layer

    Convolutional LSTM Cell.

    Parameters
    ----------
    rank : int
        1 for 1D convolution, 2 for 2D convolution, 3 for 3D convolution
    in_channels : int
        the number of input channels
    out_channels : int
        the number of output channels
    kernel_size : int, list or tuple
        the window size of convolution
    stride : int, optional
        the stride of convolution, by default 1
    padding : int, str, optional
        ``'valid'``, ``'same'``, by default ``'same'``
    dilation : int, optional
        Spacing between kernel elements. Default: 1
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default: 1
    bias : bool, optional
        If ``'True'``, adds a learnable bias to the output, by default ``'True'``
    padding_mode : str, optional
        'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    activation : str or None, optional
        activation of input convolution layer, ``'Tanh()'`` (default), ``'Sigmoid'``, ...
    rnn_activation : str or None, optional
        activation of RNN convolution layer, ``'Hardsigmoid()'`` (default), ``'Sigmoid'``, ...
    dropp : float or None, optional
        dropout rate of input convolution layer, :obj:`None`
    rnn_dropp : float or None, optional
        dropout rate of RNN layer, :obj:`None`
    device : str or None, optional
        device for computation, by default obj:`None`
    dtype : str or None, optional
        data type, by default None

    Returns
    --------
    h : Tensor
        hidden states
    c : Tensor
        code states
    
    """

    def __init__(self, rank, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', activation='Tanh()', rnn_activation='Hardsigmoid()', dropp=None, rnn_dropp=None, device=None, dtype=None):
        super(ConvLSTMCell, self).__init__()

        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype

        self.in_dropout_fn = None if dropp is None else eval('th.nn.Dropout%dd(%.4f)' % rank)
        self.rnn_dropout_fn = None if rnn_dropp is None else eval('th.nn.Dropout%dd(%.4f)' % rank)

        activation = 'Identity' if activation is None else activation

        self.activation = eval('th.nn.%s' % activation)
        self.rnn_activation = eval('th.nn.%s' % rnn_activation)

        conv_fn = eval('th.nn.Conv%dd' % rank)
        self.in_convi = conv_fn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.in_convf = conv_fn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.in_convc = conv_fn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.in_convo = conv_fn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)

        self.rnn_convi = conv_fn(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.rnn_convf = conv_fn(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.rnn_convc = conv_fn(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.rnn_convo = conv_fn(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)

    def get_hc_shape(self, xshape):
        xshape = list(xshape)
        out_size = tb.conv_size(in_size=xshape[2:], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
        hcshape = [xshape[0], self.out_channels] + out_size
        return hcshape
    
    def forward(self, x, states):

        if states is None:
            hcshape = self.get_hc_shape(x.shape)
            states = (th.zeros(hcshape, device=x.device, dtype=x.dtype), th.zeros(hcshape, device=x.device, dtype=x.dtype))
        else:
            if states[0] is None:
                hcshape = self.get_hc_shape(x.shape)
                states = list(states) if type(states) is tuple else states
                states[0] = th.zeros(hcshape, device=x.device, dtype=x.dtype)
            if states[1] is None:
                hcshape = self.get_hc_shape(x.shape)
                states = list(states) if type(states) is tuple else states
                states[1] = th.zeros(hcshape, device=x.device, dtype=x.dtype)
        h, c = states

        if self.in_dropout_fn is not None:
            xi, xf, xc, xo = self.in_dropout_fn(x), self.in_dropout_fn(x), self.in_dropout_fn(x), self.in_dropout_fn(x)
        else:
            xi, xf, xc, xo = x, x, x, x

        if self.rnn_dropout_fn is not None:
            hi, hf, hc, ho = self.rnn_dropout_fn(h), self.rnn_dropout_fn(h), self.rnn_dropout_fn(h), self.rnn_dropout_fn(h)
        else:
            hi, hf, hc, ho = h, h, h, h

        xi = self.in_convi(xi)
        xf = self.in_convf(xf)
        xc = self.in_convc(xc)
        xo = self.in_convo(xo)
        hi = self.rnn_convi(hi)
        hf = self.rnn_convf(hf)
        hc = self.rnn_convc(hc)
        ho = self.rnn_convo(ho)

        i = self.rnn_activation(xi + hi)
        f = self.rnn_activation(xf + hf)
        c = f * c + i * self.activation(xc + hc)
        o = self.rnn_activation(xo + ho)
        h = o * self.activation(c)

        return h, c


class ConvLSTM(th.nn.Module):
    r"""class for the ConvLSTM layer

    Convolutional LSTM.
    
    input shape: (B, T, C, L) or (B, T, C, H, W) or (B, T, C, H, W, K)

    Parameters
    ----------
    rank : int
        1 for 1D convolution, 2 for 2D convolution, 3 for 3D convolution
    in_channels : int, list or tuple
        the number of input channels of each cell
    out_channels : int, list or tuple
        the number of output channels of each cell
    kernel_size : int, list or tuple
        the window size of convolution of each cell
    stride : int, list or tuple, optional
        the stride of convolution of each cell, by default 1
    padding : int, str, list or tuple, optional
        the padding size of convolution of each cell, ``'valid'``, ``'same'``, by default ``'same'``
    dilation : int, list or tuple, optional
        the spacing between kernel elements of each cell, by default 1
    groups : int, list or tuple, optional
        the number of blocked connections from input channels to output channels of each cell, by default 1
    bias : bool, list or tuple, optional
        If :obj:`True`, adds a learnable bias to the output of convolution of each cell, by default :obj:`True`
    padding_mode : str, list or tuple, optional
        'zeros', 'reflect', 'replicate' or 'circular', by default 'zeros'
    activation : str or None, optional
        activation of input convolution layers, ``'Tanh()'`` (default), ``'Sigmoid'``, ...
    rnn_activation : str or None, optional
        activation of RNN convolution layers, ``'Hardsigmoid()'`` (default), ``'Sigmoid'``, ...
    dropp : float or None, optional
        dropout rate of input convolution layers, :obj:`None`
    rnn_dropp : float or None, optional
        dropout rate of RNN layers, :obj:`None`
    bidirectional : bool, optional
        :obj:`True` for bidirectional convolutional LSTM, by default :obj:`False`
    batch_first : bool, optional
        :obj:`True` for ``(B, T, ...)``, by default :obj:`False`
    device : str or None, optional
        device for computation, by default None
    dtype : str or None, optional
        data type, by default None

    Returns
    ----------
    xs : Tensor
        output sequence
    states : tuple of list
        (hidden states, code states) of each cell

    Examples
    ----------

    Stack two Conv2dLSTM Cells with ConvLSTM and ConvLSTMCell, respectively.
    
    ::

        import torch as th
        import torchbox as tb


        tb.setseed(seed=2023, target='torch')

        T, B, C, H, W = 10, 6, 2, 18, 18
        x = th.randn(T, B, C, H, W)

        # ===way1
        tb.setseed(seed=2023, target='torch')
        lstm = tb.ConvLSTM(rank=2, in_channels=[C, 4], out_channels=[4, 4], kernel_size=[3, 3], stride=[1, 1], padding=['same', 'same'])
        print(lstm.cells[0].in_convc.weight.sum(), lstm.cells[0].rnn_convc.weight.sum())
        print(lstm.cells[1].in_convc.weight.sum(), lstm.cells[1].rnn_convc.weight.sum())

        # ===way2
        tb.setseed(seed=2023, target='torch')
        cell1 = tb.ConvLSTMCell(rank=2, in_channels=C, out_channels=4, kernel_size=3, stride=1, padding='same')
        cell2 = tb.ConvLSTMCell(rank=2, in_channels=4, out_channels=4, kernel_size=3, stride=1, padding='same')

        print(cell1.in_convc.weight.sum(), cell1.rnn_convc.weight.sum())
        print(cell2.in_convc.weight.sum(), cell2.rnn_convc.weight.sum())

        # ===way1
        y, (h, c) = lstm(x, None)
        h = th.stack(h, dim=0)
        c = th.stack(c, dim=0)

        print(y.shape, y.sum(), h.shape, h.sum(), c.shape, c.sum())

        # ===way2
        h1, c1 = None, None
        h2, c2 = None, None
        y = []
        for t in range(x.shape[0]):
            h1, c1 = cell1(x[t, ...], (h1, c1))
            h2, c2 = cell2(h1, (h2, c2))
            y.append(h2)
        y = th.stack(y, dim=0)
        h = th.stack((h1, h2), dim=0)
        c = th.stack((c1, c2), dim=0)
        print(y.shape, y.sum(), h.shape, h.sum(), c.shape, c.sum())


        # output
        tensor(-1.4177, grad_fn=<SumBackward0>) tensor(0.9743, grad_fn=<SumBackward0>)
        tensor(0.1532, grad_fn=<SumBackward0>) tensor(-0.1598, grad_fn=<SumBackward0>)
        tensor(-1.4177, grad_fn=<SumBackward0>) tensor(0.9743, grad_fn=<SumBackward0>)
        tensor(0.1532, grad_fn=<SumBackward0>) tensor(-0.1598, grad_fn=<SumBackward0>)
        torch.Size([10, 6, 4, 18, 18]) tensor(-2144.8628, grad_fn=<SumBackward0>) torch.Size([2, 6, 4, 18, 18]) tensor(-398.1468, grad_fn=<SumBackward0>) torch.Size([2, 6, 4, 18, 18]) tensor(-783.8212, grad_fn=<SumBackward0>)
        torch.Size([10, 6, 4, 18, 18]) tensor(-2144.8628, grad_fn=<SumBackward0>) torch.Size([2, 6, 4, 18, 18]) tensor(-398.1468, grad_fn=<SumBackward0>) torch.Size([2, 6, 4, 18, 18]) tensor(-783.8212, grad_fn=<SumBackward0>)
    
    """

    def __init__(self, rank, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', activation='Tanh()', rnn_activation='Hardsigmoid()', dropp=None, rnn_dropp=None, bidirectional=False, batch_first=False, device=None, dtype=None):
        super(ConvLSTM, self).__init__()

        self.rank = rank

        self.in_channels = [in_channels] if (type(in_channels) is not list) and (type(in_channels) is not tuple) else in_channels
        self.out_channels = [out_channels] if (type(out_channels) is not list) and (type(out_channels) is not tuple) else out_channels
        self.ncells = len(self.in_channels)
        
        self.kernel_size = [kernel_size]*self.ncells if (type(kernel_size) is not list) and (type(kernel_size) is not tuple) else [kernel_size]*self.ncells if self.ncells != len(kernel_size) else kernel_size
        self.stride = [stride]*self.ncells if (type(stride) is not list) and (type(stride) is not tuple) else [stride]*self.ncells if self.ncells != len(stride) else stride
        self.padding = [padding]*self.ncells if (type(padding) is not list) and (type(padding) is not tuple) else [padding]*self.ncells if self.ncells != len(padding) else padding
        self.dilation = [dilation]*self.ncells if (type(dilation) is not list) and (type(dilation) is not tuple) else [dilation]*self.ncells if self.ncells != len(dilation) else dilation
        self.groups = [groups]*self.ncells if (type(groups) is not list) and (type(groups) is not tuple) else groups
        self.bias = [bias]*self.ncells if (type(bias) is not list) and (type(bias) is not tuple) else bias
        self.padding_mode = [padding_mode]*self.ncells if (type(padding_mode) is not list) and (type(padding_mode) is not tuple) else padding_mode
        self.activation = [activation]*self.ncells if (type(activation) is not list) and (type(activation) is not tuple) else activation
        self.rnn_activation = [rnn_activation]*self.ncells if (type(rnn_activation) is not list) and (type(rnn_activation) is not tuple) else rnn_activation
        self.dropp = [dropp]*self.ncells if (type(dropp) is not list) and (type(dropp) is not tuple) else dropp
        self.rnn_dropp = [rnn_dropp]*self.ncells if (type(rnn_dropp) is not list) and (type(rnn_dropp) is not tuple) else rnn_dropp
        self.batch_first = batch_first

        if bidirectional is True:
            raise ValueError('bidirectional is not supported!')

        self.bidirectional = bidirectional
        self.device = device
        self.dtype = dtype

        self.cells = []
        for i in range(self.ncells):
            self.cells.append(ConvLSTMCell(rank=rank, in_channels=self.in_channels[i], out_channels=self.out_channels[i], kernel_size=self.kernel_size[i], stride=self.stride[i], padding=self.padding[i], dilation=self.dilation[i], groups=self.groups[i], bias=self.bias[i], padding_mode=self.padding_mode[i], activation=self.activation[i], rnn_activation=self.rnn_activation[i], dropp=self.dropp[i], rnn_dropp=self.rnn_dropp[i], device=device, dtype=dtype))
        self.cells = th.nn.ModuleList(self.cells)

    def get_hc_shape(self, xshape):
        xshape = list(xshape)
        hcshape = []
        for n in range(self.ncells):
            out_size = tb.conv_size(in_size=xshape[3:] if n==0 else out_size, kernel_size=self.kernel_size[n], stride=self.stride[n], padding=self.padding[n], dilation=self.dilation[n])
            hcshape.append([xshape[1], self.out_channels[n]] + out_size)
        return hcshape

    def forward(self, x, states=None):

        if self.batch_first:
            x = x.transpose(0, 1)

        if states is None:
            hcshape = self.get_hc_shape(x.shape)
            states = [(th.zeros(hcshape[n], device=x.device, dtype=x.dtype), th.zeros(hcshape[n], device=x.device, dtype=x.dtype)) for n in range(self.ncells)]

        Ts = x.shape[0]
        xs, hs, cs = [], [states[n][0] for n in range(self.ncells)], [states[n][1] for n in range(self.ncells)]
        for t in range(0, Ts):
            hs[0], cs[0] = self.cells[0](x[t], (hs[0], cs[0]))
            for n in range(1, self.ncells):
                hs[n], cs[n] = self.cells[n](hs[n-1], (hs[n], cs[n]))
            xs.append(hs[-1])
        xs = th.stack(xs, dim=0)

        if self.batch_first:
            xs = xs.transpose(0, 1)

        return xs, (hs, cs)


if __name__ == '__main__':

    tb.setseed(seed=2023, target='torch')

    T, B, C, H, W = 10, 6, 2, 18, 18
    x = th.randn(T, B, C, H, W)

    # ===way1
    tb.setseed(seed=2023, target='torch')
    lstm = ConvLSTM(rank=2, in_channels=[C, 4], out_channels=[4, 4], kernel_size=[3, 3], stride=[1, 1], padding=['same', 'same'])
    print(lstm.cells[0].in_convc.weight.sum(), lstm.cells[0].rnn_convc.weight.sum())
    print(lstm.cells[1].in_convc.weight.sum(), lstm.cells[1].rnn_convc.weight.sum())

    # ===way2
    tb.setseed(seed=2023, target='torch')
    cell1 = ConvLSTMCell(rank=2, in_channels=C, out_channels=4, kernel_size=3, stride=1, padding='same')
    cell2 = ConvLSTMCell(rank=2, in_channels=4, out_channels=4, kernel_size=3, stride=1, padding='same')

    print(cell1.in_convc.weight.sum(), cell1.rnn_convc.weight.sum())
    print(cell2.in_convc.weight.sum(), cell2.rnn_convc.weight.sum())

    # ===way1
    y, (h, c) = lstm(x, None)
    h = th.stack(h, dim=0)
    c = th.stack(c, dim=0)

    print(y.shape, y.sum(), h.shape, h.sum(), c.shape, c.sum())

    # ===way2
    h1, c1 = None, None
    h2, c2 = None, None
    y = []
    for t in range(x.shape[0]):
        h1, c1 = cell1(x[t, ...], (h1, c1))
        h2, c2 = cell2(h1, (h2, c2))
        y.append(h2)
    y = th.stack(y, dim=0)
    h = th.stack((h1, h2), dim=0)
    c = th.stack((c1, c2), dim=0)
    print(y.shape, y.sum(), h.shape, h.sum(), c.shape, c.sum())
