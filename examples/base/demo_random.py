#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#

import torchbox as tb
import matplotlib.pyplot as plt

N, H, W = 32, 512, 512

y1 = tb.randperm(0, H, N)
x1 = tb.randperm(0, W, N)
print(len(y1), len(x1))

y2 = tb.randgrid(0, H, 32, 0., N)
x2 = tb.randgrid(0, W, 32, 0., N)
print(len(y2), len(x2))
print(y2, x2)

y3, x3 = tb.randperm([0, 0], [H, W], N)
print(len(y3), len(x3))

y4, x4 = tb.randgrid([0, 0], [H, W], [32, 32], [0.25, 0.25], N)
print(len(y4), len(x4))

plt.figure()
plt.subplot(221)
plt.grid()
plt.plot(x1, y1, '*')
plt.subplot(222)
plt.grid()
plt.plot(x2, y2, '*')
plt.subplot(223)
plt.grid()
plt.plot(x3, y3, '*')
plt.subplot(224)
plt.grid()
plt.plot(x4, y4, '*')
plt.show()







