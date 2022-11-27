#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : loss_log.py
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

    def plot(self, x=None):
        legend = []

        if self.plotdir is not None:
            self.mpl.use('Agg')

        self.plt.figure()

        for k, v in self.losses.items():
            if len(v) > 0:
                if x is None:
                    self.plt.plot(v)
                else:
                    self.plt.plot(x, v)
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

