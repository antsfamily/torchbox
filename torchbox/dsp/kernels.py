#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : kernels.py
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
