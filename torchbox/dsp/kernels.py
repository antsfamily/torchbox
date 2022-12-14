#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : kernels.py
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


BOX_BLUR_3X3 = th.full((3, 3), 1.0 / 9.0, dtype=th.float32)

BOX_BLUR_5X5 = th.full((5, 5), 1.0 / 25.0, dtype=th.float32)

GAUSSIAN_BLUR_3x3 = (1.0 / 16.0) * th.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=th.float32)

VERTICAL_SOBEL_3x3 = th.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=th.float32)

HORIZONTAL_SOBEL_3x3 = th.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=th.float32)


if __name__ == '__main__':

    print(BOX_BLUR_3X3)
    print(BOX_BLUR_5X5)
    print(GAUSSIAN_BLUR_3x3)
    print(VERTICAL_SOBEL_3x3)
    print(HORIZONTAL_SOBEL_3x3)
