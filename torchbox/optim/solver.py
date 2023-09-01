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

import time
import torch as th
import torchbox as tb


def train_epoch(model, dl, criterions, criterionws=None, optimizer=None, scheduler=None, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    r"""train one epoch

    Parameters
    ----------
    model : Module
        an instance of torch.nn.Module
    dl : DataLoader
        the dataloader for training
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    optimizer : Optimizer or None
        an instance of torch.optim.Optimizer, default is :obj:`None`, 
        which means ``th.optim.Adam(model.parameters(), lr=0.001)``
    scheduler : LrScheduler or None
        an instance of torch.optim.LrScheduler, default is :obj:`None`, 
        which means using fixed learning rate
    epoch : int
        epoch index
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for training, by default ``'cuda:0'``
    kwargs :
        other forward args

    see also :func:`~torchbox.optim.solver.valid_epoch`, :func:`~torchbox.optim.solver.test_epoch`, :func:`~torchbox.optim.save_load.save_model`, :func:`~torchbox.optim.save_load.load_model`.
        
    """

    model.train()
    logf = None if logf == 'terminal' else logf
    criterionws = [1.] * len(criterions) if criterionws is None else criterionws
    optimizer = th.optim.Adam(model.parameters(), lr=0.001) if optimizer is None else optimizer
    
    tstart = time.time()
    lossv = 0.
    for batch_idx, (data, target) in enumerate(dl):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model.forward(data, **kwargs)

        loss = 0.
        for criterionw, criterion in zip(criterionws, criterions):
            loss += criterionw * criterion(output, target)
        
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        lossv += loss.item()
    lossv /= len(dl.dataset)
    tend = time.time()
    if epoch is None:
        print('--->Valid, loss: %.4f, time: %.2f' % (lossv, tend - tstart), file=logf)
    else:
        print('--->Valid epoch %d, loss: %.4f, time: %.2f' % (epoch, lossv, tend - tstart), file=logf)
    return lossv

def valid_epoch(model, dl, criterions, criterionws=None, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    r"""valid one epoch

    Parameters
    ----------
    model : function handle
        an instance of torch.nn.Module
    dl : dataloder
        the validation dataloader
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    epoch : int
        epoch index,  default is None
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for validation, by default ``'cuda:0'``
    kwargs :
        other forward args

    see also :func:`~torchbox.optim.solver.train_epoch`, :func:`~torchbox.optim.solver.test_epoch`, :func:`~torchbox.optim.save_load.save_model`, :func:`~torchbox.optim.save_load.load_model`.

    """

    model.eval()
    logf = None if logf == 'terminal' else logf
    criterionws = [1.] * len(criterions) if criterionws is None else criterionws

    tstart = time.time()
    lossv = 0.
    with th.no_grad():
        for batch_idx, (data, target) in enumerate(dl):
            data, target = data.to(device), target.to(device)

            output = model.forward(data, **kwargs)

            loss = 0.
            for criterionw, criterion in zip(criterionws, criterions):
                loss += criterionw * criterion(output, target)
            
            lossv += loss.item()
    lossv /= len(dl.dataset)
    tend = time.time()
    if epoch is None:
        print('--->Valid, loss: %.4f, time: %.2f' % (lossv, tend - tstart), file=logf)
    else:
        print('--->Valid epoch %d, loss: %.4f, time: %.2f' % (epoch, lossv, tend - tstart), file=logf)
    return lossv

def test_epoch(model, dl, criterions, criterionws=None, epoch=None, logf='terminal', device='cuda:0', **kwargs):
    """Test one epoch

    Parameters
    ----------
    model : function handle
        an instance of torch.nn.Module
    dl : dataloder
        the testing dataloader
    criterions : list or tuple
        list of loss function
    criterionws : list or tuple
        list of float loss weight
    epoch : int or None
        epoch index,  default is None
    logf : str or object, optional
        IO for print log, file path or ``'terminal'`` (default)
    device : str, optional
        device for testing, by default ``'cuda:0'``
    kwargs :
        other forward args

    see also :func:`~torchbox.optim.solver.train_epoch`, :func:`~torchbox.optim.solver.valid_epoch`, :func:`~torchbox.optim.save_load.save_model`, :func:`~torchbox.optim.save_load.load_model`.

    """

    model.eval()
    logf = None if logf == 'terminal' else logf
    criterionws = [1.] * len(criterions) if criterionws is None else criterionws

    tstart = time.time()
    lossv = 0.
    outputs = []
    with th.no_grad():
        for batch_idx, (data, target) in enumerate(dl):
            data, target = data.to(device), target.to(device)

            output = model.forward(data, **kwargs)
            outputs.append(output.detach().cpu())

            loss = 0.
            for criterionw, criterion in zip(criterionws, criterions):
                loss += criterionw * criterion(output, target)
            
            lossv += loss.item()
    lossv /= len(dl.dataset)
    tend = time.time()
    if epoch is None:
        print('--->Test, loss: %.4f, time: %.2f' % (lossv, tend - tstart), file=logf)
    else:
        print('--->Test epoch %d, loss: %.4f, time: %.2f' % (epoch, lossv, tend - tstart), file=logf)
    outputs = th.cat(outputs, dim=0)
    return lossv, outputs

