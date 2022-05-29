#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import torch as th
import torch.nn as nn


conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0,
                  dilation=1, groups=1, bias=True, padding_mode='zeros')

print(conv1)

Ci = 4
Co = 16

# exact output size can be also specified as an argument
input = th.randn(3, Ci, 12, 12)
print(input.size())
downsample = nn.Conv2d(Ci, Co, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(Co, Ci, 3, stride=2, padding=1)
h = downsample(input)
print(h.size())

output = upsample(h)
# output = upsample(h, output_size=input.size())
print(output.size())


pool = nn.MaxPool2d(2, stride=1, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=1)
input = th.randn(3, Ci, 12, 12)
print(input.size(), "input")
output, indices = pool(input)
print(output.size())
xx = unpool(output, indices)
print(xx.size())

# specify a different output size than input size
xx = unpool(output, indices, output_size=th.Size([3, Ci, 12, 12]))

print(xx.size())
