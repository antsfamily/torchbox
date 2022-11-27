#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : lr_scheduler.py
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

import math
import warnings
import torch as th


class GaussianLR(th.optim.lr_scheduler._LRScheduler):
    r"""GaussianLR

    Set the learning rate of each parameter group using a double gaussian kernel schedule

    .. image:: ./_static/GaussianLREquation.png
       :scale: 50 %
       :align: center

    where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators.

    The maximum learning rate are the base learning rate setted in Optimizer.


    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    t_eta_max : int
        Iterations when the learning rate reach to the maximum value :math:`\eta_{\max}`.
    sigma1 : int
        Controls the shape of warming up phase.
    sigma2 : int
        Controls the shape of annealing phase.
    eta_start : float
        Starting learning rate. Default: 0.
    eta_stop : float
        Stopping learning rate. Default: 0.
    last_epoch : int
        The index of last epoch. Default: -1.

    Examples
    ---------

    .. image:: ./_static/DoubleGaussianKernelLR.png
       :scale: 50 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torch as th
        import torchbox as tb
        import matplotlib; matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        lr = 1e-1
        lr = 1e-2
        # lr = 1e2

        num_epochs = 1000
        num_epochs = 500
        batch_size = 8
        num_batch = 750

        params = {th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
                th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
                }

        optimizer = th.optim.Adam(params, lr=lr)
        # optimizer = th.optim.SGD(params, lr=lr, momentum=0.9)
        scheduler = tb.optim.lr_scheduler.GaussianLR(optimizer, t_eta_max=50, sigma1=15, sigma2=100, eta_start=1e-4, eta_stop=1e-3, last_epoch=-1)

        print(optimizer)

        lrs = []
        for n in range(num_epochs):
            for b in range(num_batch):

                optimizer.step()

                # lrs.append(optimizer.param_groups[0]['lr'])

            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        plt.figure()
        plt.plot(lrs)
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.grid()
        plt.show()

    """

    def __init__(self, optimizer, t_eta_max, sigma1, sigma2, eta_start=1e-6, eta_stop=1e-5, last_epoch=-1):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.t_eta_max = t_eta_max
        self.eta_start = eta_start
        self.eta_stop = eta_stop
        # self.eta_max = eta_max  # base lr
        super(GaussianLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch == 0:
            return [self.eta_start for base_lr in self.base_lrs]
        elif self.last_epoch < self.t_eta_max:
            return [group['lr'] + (base_lr - self.eta_start) * (math.exp(-(self.last_epoch + 1 - self.t_eta_max)**2 / 2. / (self.sigma1**2)) - math.exp(-(self.last_epoch - self.t_eta_max)**2 / 2. / (self.sigma1**2)))
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [group['lr'] + (base_lr - self.eta_stop) * (math.exp(-(self.last_epoch + 1 - self.t_eta_max)**2 / 2. / (self.sigma2**2)) - math.exp(-(self.last_epoch - self.t_eta_max)**2 / 2. / (self.sigma2**2)))
                for base_lr, group in
                zip(self.base_lrs, self.optimizer.param_groups)]

    def _get_closed_form_lr(self):
        if self.last_epoch < self.t_eta_max:
            return [(base_lr - self.eta_start) * math.exp(-(self.last_epoch - self.t_eta_max)**2 / 2. / (self.sigma1**2)) + self.eta_start
                    for base_lr in self.base_lrs]
        else:
            return [(base_lr - self.eta_stop) * math.exp(-(self.last_epoch - self.t_eta_max)**2 / 2. / (self.sigma2**2)) + self.eta_stop
                    for base_lr in self.base_lrs]


class MountainLR(th.optim.lr_scheduler._LRScheduler):
    r"""MountainLR
    
    Set the learning rate of each parameter group using a double gaussian kernel

    .. math::
        (|x-P| / N) .* (-2 + cos(2 * (x-P) / T))

    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators.

    The maximum learning rate are the base learning rate setted in Optimizer.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    t_eta_max : int
        Iterations when the learning rate reach to the maximum value :math:`\eta_{\max}`.
    sigma1 : int
        Controls the shape of warming up phase.
    sigma2 : int
        Controls the shape of annealing phase.
    eta_start : float
        Starting learning rate. Default: 0.
    eta_stop : float
        Stopping learning rate. Default: 0.
    last_epoch : int
        The index of last epoch. Default: -1.

    Examples
    ---------

    .. image:: ./_static/MountainLR.png
       :scale: 50 %
       :align: center

    The results shown in the above figure can be obtained by the following codes.

    ::

        import torch as th
        import torchbox as tb
        import matplotlib; matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        lr = 1e-1
        lr = 1e-2
        # lr = 1e2

        num_epochs = 1000
        num_epochs = 500
        batch_size = 8
        num_batch = 750

        params = {th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
                th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
                }

        optimizer = th.optim.Adam(params, lr=lr)
        scheduler = tb.optim.lr_scheduler.MountainLR(optimizer, total_epoch=num_epochs, peak_epoch=300, period_epoch=50, last_epoch=-1)

        print(optimizer)

        lrs = []
        for n in range(num_epochs):
            for b in range(num_batch):

                optimizer.step()

                # lrs.append(optimizer.param_groups[0]['lr'])

            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        plt.figure()
        plt.plot(lrs)
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.grid()
        plt.show()

    """

    def __init__(self, optimizer, total_epoch, peak_epoch, period_epoch, last_epoch=-1):
        self.total_epoch = total_epoch
        self.peak_epoch = peak_epoch
        self.period_epoch = period_epoch
        super(MountainLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch == 0:
            lr = 10.**((self.peak_epoch * 1. / self.total_epoch) * (-2. + math.cos(2 * math.pi * (-self.peak_epoch) / self.period_epoch)))
            return [base_lr * lr for base_lr in self.base_lrs]
        lr = 10.**((abs(self.last_epoch + 1 - self.peak_epoch) * 1. / self.total_epoch) * (-2. + math.cos(2 * math.pi * (self.last_epoch + 1 - self.peak_epoch) / self.period_epoch))) - 10.**((abs(self.last_epoch - self.peak_epoch) * 1. / self.total_epoch) * (-2. + math.cos(2 * math.pi * (self.last_epoch - self.peak_epoch) / self.period_epoch)))
        return [group['lr'] + base_lr * lr
                for base_lr, group in
                zip(self.base_lrs, self.optimizer.param_groups)]

    def _get_closed_form_lr(self):
        lr = 10.**((abs(self.last_epoch - self.peak_epoch) * 1. / self.total_epoch) * (-2. + math.cos(2 * math.pi * (self.last_epoch - self.peak_epoch) / self.period_epoch)))
        return [base_lr * lr for base_lr in self.base_lrs]


if __name__ == '__main__':

    import torch as th
    import torchbox as tb
    import matplotlib; matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    lr = 1e-1
    lr = 1e-2
    # lr = 1e2

    num_epochs = 1000
    num_epochs = 500
    batch_size = 8
    num_batch = 750

    params = {th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
              th.nn.parameter.Parameter(th.zeros(128), requires_grad=True),
              }

    optimizer = th.optim.Adam(params, lr=lr)
    # optimizer = th.optim.SGD(params, lr=lr, momentum=0.9)
    # scheduler = tb.optim.lr_scheduler.GaussianLR(optimizer, t_eta_max=80, sigma1=15, sigma2=200, eta_start=1e-4, eta_stop=1e-5, last_epoch=-1)
    # scheduler = tb.optim.lr_scheduler.GaussianLR(optimizer, t_eta_max=50, sigma1=15, sigma2=100, eta_start=1e-4, eta_stop=1e-3, last_epoch=-1)
    scheduler = tb.optim.lr_scheduler.MountainLR(optimizer, total_epoch=num_epochs, peak_epoch=300, period_epoch=50, last_epoch=-1)

    print(optimizer)

    lrs = []
    for n in range(num_epochs):
        for b in range(num_batch):

            optimizer.step()

            # lrs.append(optimizer.param_groups[0]['lr'])

        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    plt.figure()
    plt.plot(lrs)
    plt.xlabel('Iteration')
    plt.ylabel('Learning rate')
    plt.grid()
    plt.show()
