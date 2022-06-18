#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
#

import torchbox as tb


datafolder = tb.data_path('optical')
x = tb.imread(datafolder + 'LenaRGB512.tif')
print(x.shape)

# rects, edgecolors, fillcolors, linewidths = [[0, 0, 511, 511]], [None], [[0, 255, 0]], [1]
# rects, edgecolors, fillcolors, linewidths = [[0, 0, 511, 511]], [[255, 0, 0]], [None], [1]
# rects, edgecolors, fillcolors, linewidths = [[0, 0, 511, 511]], [[255, 0, 0]], [[0, 255, 0]], [1]
rects, edgecolors, fillcolors, linewidths = [[64, 64, 128, 128], [200, 200, 280, 400]], [[0, 255, 0], [0, 0, 255]], [None, [255, 255, 0]], [1, 6]

y = tb.draw_rectangle(x, rects, edgecolors=edgecolors, linewidths=linewidths, fillcolors=fillcolors, axes=[(0, 1)])

tb.imsave('out.png', y)

plt = tb.imshow([x, y], titles=['original', 'drew'])
plt.show()

