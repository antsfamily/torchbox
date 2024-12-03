#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-10-15 10:34:16
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import torch as th
import torchbox as tb

X = th.randn(1, 3, 4, 2)
S = tb.entropy(X, cdim=-1, dim=None, mode='shannon')
print(S)

X = X[:, :, :, 0] + 1j * X[:, :, :, 1]
S = tb.entropy(X, cdim=None, dim=None, mode='shannon')
print(S)
