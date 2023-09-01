#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : demo_convlstm.py
# @author    : Zhi Liu
# @email     : zhiliu.mind@gmail.com
# @homepage  : http://iridescent.ink
# @date      : Thu Jul 20 2022
# @version   : 0.0
# @license   : GNU General Public License (GPL)
# @note      : 
# 
# The GNU General Public License (GPL)
# Copyright (C) 2013- Zhi Liu
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import torch as th
import torchbox as tb


class MetaDataLoader:

    def __init__(self, x, y, bstask, nway=1, kshot=1, kquery=1, sfltask=False, sflpoint=False, dsname='train'):
        self.x = x  # nTask, nway, kpoint>=(kshot+kquery), ...
        self.y = y
        self.bstask = bstask
        self.nway = nway
        self.kshot = kshot
        self.kquery = kquery
        self.sfltask = sfltask
        self.sflpoint = sflpoint
        self.dsname = dsname
        self.ntask = x.shape[0]
        self.kpoint = x.shape[2]
        self.idxtask = 0
        self.indextask = th.arange(0, self.ntask)
        self.indexpoint = th.arange(0, self.kpoint)

    def next(self):

        if self.idxtask >= self.ntask:
            self.idxtask = 0
            if self.sfltask:
                self.indextask = tb.randperm(0, self.ntask, self.ntask)
            else:
                self.indextask = th.arange(0, self.ntask)
            if self.sflpoint:
                self.indexpoint = tb.randperm(0, self.kpoint, self.kpoint)
            else:
                self.indexpoint = th.arange(0, self.kpoint)
        xspt = self.x[self.indextask[self.idxtask:self.idxtask+self.bstask]][:, :, self.indexpoint[:self.kshot]]
        yspt = self.y[self.indextask[self.idxtask:self.idxtask+self.bstask]][:, :, self.indexpoint[:self.kshot]]
        xqry = self.x[self.indextask[self.idxtask:self.idxtask+self.bstask]][:, :, self.indexpoint[self.kshot:self.kshot+self.kquery]]
        yqry = self.y[self.indextask[self.idxtask:self.idxtask+self.bstask]][:, :, self.indexpoint[self.kshot:self.kshot+self.kquery]]
        self.idxtask += self.bstask
        return xspt, yspt, xqry, yqry

    def __len__(self):

        nbs = self.ntask // self.bstask if self.ntask % self.bstask == 0 else self.ntask // self.bstask + 1
        return nbs


if __name__ == '__main__':

    x = th.rand(100, 5, 20, 6, 64, 64)
    y = th.rand(100, 5, 20, 2, 64, 64)


    dltrain = MetaDataLoader(x, y, bstask=32, nway=5, kshot=10, kquery=6, sfltask=True, sflpoint=True, dsname='train')

    print(len(dltrain))
    for b in range(len(dltrain)):
        xspt, yspt, xqry, yqry = dltrain.next()
        print(b, xspt.shape, yspt.shape, xqry.shape, yqry.shape)


