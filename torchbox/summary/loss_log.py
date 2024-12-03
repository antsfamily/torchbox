#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : loss_log.py
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

class LossLog():

    def __init__(self, plotdir=None, xlabel='Epoch', ylabel='Loss', title=None, filename=None, logdict=None, lom='min'):
        self.plotdir = plotdir
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.filename = filename
        self.lom = lom
        if logdict is None:
            self.losses = {'train': [], 'valid': [], 'test': []}
        else:
            self.losses = logdict

        self.bests = {}
        for k, v in self.losses.items():
            if len(v) < 1:
                self.bests[k] = float('inf') if self.lom in ['min', 'MIN'] else float('-inf')
            else:
                self.bests[k] = eval(self.lom)(v)

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        self.mpl = mpl
        self.plt = plt
        self.backend = mpl.get_backend()

    def assign(self, key, value):
        self.losses[key] = value

    def add(self, key, value):
        self.losses[key].append(value)

    def get(self, key=None):
        return self.losses[key]

    def updir(self, plotdir=None):
        self.plotdir = plotdir

    def plot(self, x=None, offset=0):
        legend = []

        if self.plotdir is not None:
            self.mpl.use('Agg')

        self.plt.figure()

        for k, v in self.losses.items():
            if len(v) > 0:
                if x is None:
                    self.plt.plot(v[offset:])
                else:
                    self.plt.plot(x[offset:], v)
                legend.append(k)
        self.plt.legend(legend)
        self.plt.xlabel(self.xlabel)
        self.plt.ylabel(self.ylabel)
        self.plt.grid()

        if self.title is not None:
            self.plt.title(self.title)

        if self.plotdir is None:
            self.plt.show()
        else:
            if self.filename is None:
                self.plt.savefig(self.plotdir + '/' + self.ylabel + '_' + self.xlabel + '.png')
            else:
                self.plt.savefig(self.plotdir + '/' + self.filename)
            self.plt.close()
            self.mpl.use(self.backend)
        
    def judge(self, key, n1=50, n2=10):
        r"""judge how to save weights

        |____n1____|__n2__||
                       current epoch

        If the average loss of the last n2 epochs is better than the average of the previous n1 epochs 
        and the loss value of the current epoch is the best among the n2 epochs, 
        then save the weights of current epoch with ``'Average'`` flag. 
        If the loss of current epoch is the best of all previous epochs, then save the weights of 
        current epoch with ``'Single'`` flag. 

        Parameters
        ----------
        key : str
            the key string of loss for judge.
        n1 : int, optional
            the number of latest epochs, by default 50
        n2 : int, optional
            the number of , by default 10

        Returns
        -------
        bool
            The current is the best?
        str
            ``'Single'`` or ``'Average'``
        """
        loss = self.losses[key]
        n = len(loss)
        flag, proof = False, ''

        if self.lom in ['min', 'MIN']:
            if loss[-1] < self.bests[key]:
                self.bests[key] = loss[-1]
                flag = True
                proof += 'Single'
            if n > n1 + n2:
                if (sum(loss[-n2:]) / n2 <= sum(loss[-n2 - n1:-n2]) / n1) and (loss[-n2:].index(min(loss[-n2:])) == n2 - 1):
                    flag = True
                    proof += 'Average'

        return flag, proof


if __name__ == '__main__':

    loslog = LossLog(plotdir='./', xlabel='xlabel', ylabel='ylabel')
    loslog = LossLog(plotdir=None, xlabel='Epoch', ylabel='Loss', title=None, filename='LossEpoch', logdict={'train': [], 'valid': []})
    for n in range(100):
        loslog.add('train', n)
        loslog.add('valid', n - 1)

    loslog.plot()

