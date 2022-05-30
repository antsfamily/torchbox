#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from matplotlib import pyplot as plt


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
        try:
            global plt
            plt.figure()
        except Exception as e:
            # print(e.args)
            # print(str(e))
            # print(repr(e))
            import matplotlib; matplotlib.use('Agg');
            import matplotlib.pyplot as plt
            plt.figure()

        for k, v in self.losses.items():
            if len(v) > 0:
                if x is None:
                    plt.plot(v)
                else:
                    plt.plot(x, v)
                legend.append(k)
        plt.legend(legend)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.grid()

        if self.title is not None:
            plt.title(self.title)

        if self.plotdir is None:
            plt.show()
        else:
            if self.filename is None:
                plt.savefig(self.plotdir + '/' + self.ylabel + '_' + self.xlabel + '.png')
            else:
                plt.savefig(self.plotdir + '/' + self.filename)
            plt.close()

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
    loslog = LossLog(plotdir='./', xlabel='Epoch', ylabel='Loss', title=None, filename='LossEpoch', logdict={'train': [], 'valid': []})
    for n in range(100):
        loslog.add('train', n)
        loslog.add('valid', n - 1)

    loslog.plot()
