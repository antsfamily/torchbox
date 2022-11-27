#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : colors.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Sun Nov 27 2019
# @version   : 0.0
# @license   : The Apache License 2.0
# @note      : https://sashamaps.net/docs/resources/20-colors/
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

import torch as th
import matplotlib.colors as mcolors
from matplotlib import cm
from torchbox.utils.colormaps import cmaps
from torchbox.base.mathops import nextpow2
from torchbox.misc.transform import scale


DISTINCT_COLORS_HEX = {
    'red': '#e6194B', 'green': '#3cb44b', 'yellow': '#ffe119', 'blue': '#4363d8',
    'orange': '#f58231', 'purple': '#911eb4', 'cyan': '#42d4f4', 'magenta': '#f032e6',
    'lime': '#bfef45', 'pink': '#fabed4', 'teal': '#469990', 'lavende': '#dcbeff',
    'brown': '#9A6324', 'beige': '#fffac8', 'maroon': '#800000', 'mint': '#aaffc3',
    'olive': '#808000', 'apricot': '#ffd8b1', 'navy': '#000075', 'grey': '#a9a9a9',
    'white': '#ffffff', 'black': '#000000'
}

DISTINCT_COLORS_RGB = {
    'red': (230, 25, 75), 'green': (60, 180, 75), 'yellow': (255, 225, 25), 'blue': (0, 130, 200),
    'orange': (245, 130, 48), 'purple': (145, 30, 180), 'cyan': (70, 240, 240), 'magenta': (240, 50, 230),
    'lime': (210, 245, 60), 'pink': (250, 190, 212), 'teal': (0, 128, 128), 'lavende': (220, 190, 255),
    'brown': (170, 110, 40), 'beige': (255, 250, 200), 'maroon': (128, 0, 0), 'mint': (170, 255, 195),
    'olive': (128, 128, 0), 'apricot': (255, 215, 180), 'navy': (0, 0, 128), 'grey': (128, 128, 128),
    'white': (255, 255, 255), 'black': (0, 0, 0)
}

DISTINCT_COLORS_CMYK = {
    'red': (0, 100, 66, 0), 'green': (75, 0, 100, 0), 'yellow': (0, 25, 95, 0), 'blue': (100, 35, 0, 0),
    'orange': (0, 60, 92, 0), 'purple': (35, 70, 0, 0), 'cyan': (70, 0, 0, 0), 'magenta': (0, 100, 0, 0),
    'lime': (35, 0, 100, 0), 'pink': (0, 24, 100, 2), 'teal': (100, 0, 0, 50), 'lavende': (14, 25, 100, 0),
    'brown': (0, 35, 75, 33), 'beige': (5, 10, 30, 0), 'maroon': (0, 100, 100, 50), 'mint': (33, 0, 23, 0),
    'olive': (0, 0, 100, 50), 'apricot': (0, 15, 30, 0), 'navy': (100, 100, 0, 50), 'grey': (0, 0, 0, 50),
    'white': (0, 0, 0, 0), 'black': (0, 0, 0, 100)
}

DISTINCT_COLORS_RGB_NORM = {
    'red': (230 / 255, 25 / 255, 75 / 255), 'green': (60 / 255, 180 / 255, 75 / 255), 'yellow': (255 / 255, 225 / 255, 25 / 255), 'blue': (0 / 255, 130 / 255, 200 / 255),
    'orange': (245 / 255, 130 / 255, 48 / 255), 'purple': (145 / 255, 30 / 255, 180 / 255), 'cyan': (70 / 255, 240 / 255, 240 / 255), 'magenta': (240 / 255, 50 / 255, 230 / 255),
    'lime': (210 / 255, 245 / 255, 60 / 255), 'pink': (250 / 255, 190 / 255, 212 / 255), 'teal': (0 / 255, 128 / 255, 128 / 255), 'lavende': (220 / 255, 190 / 255, 255 / 255),
    'brown': (170 / 255, 110 / 255, 40 / 255), 'beige': (255 / 255, 250 / 255, 200 / 255), 'maroon': (128 / 255, 0 / 255, 0 / 255), 'mint': (170 / 255, 255 / 255, 195 / 255),
    'olive': (128 / 255, 128 / 255, 0 / 255), 'apricot': (255 / 255, 215 / 255, 180 / 255), 'navy': (0 / 255, 0 / 255, 128 / 255), 'grey': (128 / 255, 128 / 255, 128 / 255),
    'white': (255 / 255, 255 / 255, 255 / 255), 'black': (0 / 255, 0 / 255, 0 / 255)
}


# https://matplotlib.org/stable/gallery/color/named_colors.html

BASE_COLORS = mcolors.BASE_COLORS
TABLEAU_COLORS = mcolors.TABLEAU_COLORS
CSS4_COLORS = mcolors.CSS4_COLORS


def rgb2gray(rgb, fmt='chnllast'):
    r"""Converts RGB image to GRAY image

    Converts RGB image to GRAY image according to

    .. math::
       G = 0.2989R + 0.5870G + 0.1140B

    see matlab's ``rgb2gray`` function for details.

    Parameters
    ----------
    rgb : Tensor
        Original RGB tensor.
    fmt : str, optional
        Specifies the position of channels in :attr:`rgb` tensor, surpported are:
        - ``'chnllast'`` (default)
        - ``'chnlfirst'``
    """

    dtype = rgb.dtype

    if rgb.dim() < 3:
        return rgb

    if fmt in ['chnllast', 'ChnlLast']:
        return (0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]).to(dtype)
    if fmt in ['chnlfirst', 'ChnlFirst']:
        return (0.2989 * rgb[0, ...] + 0.5870 * rgb[1, ...] + 0.1140 * rgb[2, ...]).to(dtype)


def gray2rgb(gray, cmap, drange=[0, 255], fmtstr=False):
    r"""Converts gray images to rgb image

    converts `gray` images to rgb image according to the specified colormap string 'cmap',
    supported are: ``'parula'``, ``'jet'``, ``'hsv'``, ``'hot'``, ``'cool'``, ``'spring'``,
    ``'summer'``, ``'autumn'``, ``'winter'``, ``'gray'``, ``'bone'``, ``'copper'``, ``'pink'``,
    ``'lines'``, ``'colorcube'``, ``'prism'``, ``'flag'``, ``'white'``.

    Parameters
    ----------
    gray : torch tensor
        The gray image :math:`N\times H\times W`, or the gray images :math:`N\times H\times W`, :math:`N` is the number of images.
    cmap : str or colormap
        colormap string
    drange : list or tuple
        The low and high value, default is ``[0, 255]``
    fmtstr : bool or str
        Specifies the format (``'int'``, ``'uint8'``, ``'uint16'``, ``'uint32'``, ``'float16'``,
        ``'float32'``, , ``'float64'``) for the output rgb matrix. (default is ``False``, don't change)

    Examples
    --------

    ::

        import torchbox as tb
        import matplotlib.pyplot as plt
        import torch as th

        cmap = 'jet'
        # cmap = 'hsv'
        # cmap = 'hot'
        # cmap = 'parula'
        gray = ts.imread('../../data/images/LenaGRAY256.png')
        print(gray.shape)

        rgb = gray2rgb(gray, cmap, [0, 1], False)  # rgb --> double, [0, 1]
        # rgb = gray2rgb(gray, cmap, [0, 255], False)  # rgb --> double, [0., 255.]
        # rgb = gray2rgb(gray, cmap, [0, 255], True)  # rgb --> uint8, [0, 255]

        print(gray.shape, th.min(gray), th.max(gray), gray.dtype)
        print(rgb.shape, th.min(rgb), th.max(rgb), rgb.dtype)

        plt.figure()
        plt.subplot(121)
        plt.imshow(gray, cmap=pb.parula if cmap == 'parula' else cmap)
        plt.subplot(122)
        plt.imshow(rgb)
        plt.show()
    """

    L = 256 if (drange[1] == 1) and (drange[0] == 0) else drange[1] - drange[0] + 1
    flag = False
    if gray.ndim == 2:
        flag = True
        gray.unsqueeze(0)

    if cmap == 'parula':
        colormap = cmaps['parula']
        L = 64
    else:
        colormap = cm.get_cmap(cmap, L) if type(cmap) is str else cmap

    N, H, W = gray.shape
    gray = gray.cpu()
    rgb = th.zeros(N, H, W, 3)
    for n in range(N):
        gray[n] = scale(gray[n], st=[0, L], sf=None, istrunc=True, extra=False).int()
        rgb[n] = th.from_numpy(colormap(gray[n])[:, :, :3])

    rgb = drange[0] + (drange[1] - drange[0]) * rgb  # [drange(1), drange(2)]
    if fmtstr:
        rgb = rgb.to(dtype=fmtstr)

    return rgb.squeeze(0) if flag else rgb


if __name__ == '__main__':

    A = th.randn(5, 4, 3)
    print(A)
    print(rgb2gray(A, fmt='chnlfirst'))

    A = th.randn(3, 5, 4)
    print(A)
    print(rgb2gray(A, fmt='chnllast'))

    import torchbox as tb
    import matplotlib.pyplot as plt

    cmap = 'jet'
    # cmap = 'hsv'
    # cmap = 'hot'
    # cmap = 'parula'
    # gray = tb.imread('../../data/images/LenaGRAY256.png')
    gray = tb.imread('/mnt/e/ws/github/antsfamily/torchbox/torchbox/data/images/LenaGRAY256.png')
    print(gray.shape)

    rgb = gray2rgb(gray, cmap, [0, 1], False)  # rgb --> double, [0, 1]
    # rgb = gray2rgb(gray, cmap, [0, 255], False)  # rgb --> double, [0., 255.]
    # rgb = gray2rgb(gray, cmap, [0, 255], True)  # rgb --> uint8, [0, 255]

    print(gray.shape, th.min(gray), th.max(gray), gray.dtype)
    print(rgb.shape, th.min(rgb), th.max(rgb), rgb.dtype)

    plt.figure()
    plt.subplot(121)
    plt.imshow(gray.numpy(), cmap=tb.parula if cmap == 'parula' else cmap)
    plt.subplot(122)
    plt.imshow(rgb.numpy())
    plt.show()

