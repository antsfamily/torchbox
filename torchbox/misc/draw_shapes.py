#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-07-06 22:29:14
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import

import torch as th
from torchbox.base.arrayops import sl


def draw_rectangle(x, rects, edgecolors=[[255, 0, 0]], linewidths=[1], fillcolors=[None], axes=(-3, -2)):
    """Draw rectangles in a tensor


    Parameters
    ----------
    x : Tensor
        The input with any size.
    rects : list or tuple
        The coordinates of the rectangles [[lefttop, rightbottom]].
    edgecolors : list, optional
        The color of edge.
    linewidths : int, optional
        The linewidths of edge.
    fillcolors : int, optional
        The color for filling.
    axes : int, optional
        The axes for drawing the rect (default [(-3, -2)]).

    Returns
    ---------

    y : tensor
        The tensors with rectangles.

    Examples
    ----------
    
    .. image:: ./_static/DRAWSHAPEdemo.png
       :scale: 100 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

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


    """

    axes = axes * len(rects) if len(axes) == 1 and len(rects) > 1 else axes

    if type(x) is not th.Tensor:
        y = th.tensor(x)
    else:
        y = th.clone(x)
    
    d = y.dim()

    for rect, edgecolor, linewidth, fillcolor, axis in zip(rects, edgecolors, linewidths, fillcolors, axes):
        edgecolor = th.tensor(edgecolor, dtype=y.dtype) if edgecolor is not None else None
        fillcolor = th.tensor(fillcolor, dtype=y.dtype) if fillcolor is not None else None
        if edgecolor is not None:
            top, left, bottom, right = rect
            for l in range(linewidth):
                y[sl(d, axis, [slice(top, bottom + 1), [left, right]])] = edgecolor
                y[sl(d, axis, [[top, bottom], slice(left, right + 1)])] = edgecolor
                top += 1
                left += 1
                bottom -= 1
                right -= 1
        if fillcolor is not None:
            top, left, bottom, right = rect
            top += linewidth
            left += linewidth
            bottom -= linewidth
            right -= linewidth
            y[sl(d, axis, [slice(top, bottom + 1), slice(left, right + 1)])] = fillcolor
    return y


def draw_eclipse(x, centroids, aradii, bradii, edgecolors=[255, 0, 0], linewidths=1, fillcolors=None, axes=(-2, -1)):

    for centroid, aradius, bradius in centroids, aradii, bradii:
        pass


if __name__ == '__main__':

    x = th.zeros(2, 8, 10, 3)

    rects = [[1, 2, 6, 8]]

    y = draw_rectangle(x, rects, edgecolors=[[100, 125, 255]], linewidths=[2], fillcolors=[None], axes=[(-3, -2)])

    print(x[0, :, :, 0])
    print(x[0, :, :, 1])
    print(x[0, :, :, 2])

    print(y[0, :, :, 0])
    print(y[0, :, :, 1])
    print(y[0, :, :, 2])

    y = draw_rectangle(x, rects, edgecolors=[[100, 125, 255]], linewidths=[2], fillcolors=[[20, 50, 80]], axes=[(-3, -2)])

    print(x[0, :, :, 0])
    print(x[0, :, :, 1])
    print(x[0, :, :, 2])

    print(y[0, :, :, 0])
    print(y[0, :, :, 1])
    print(y[0, :, :, 2])
