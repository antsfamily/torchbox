#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : save_load.py
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


def device_transfer(obj, name, device):
    if name in ['optimizer', 'Optimizer']:
        for group in obj.param_groups:
            for p in group['params']:
                for k, v in obj.state[p].items():
                    if th.is_tensor(v):
                        obj.state[p][k] = v.to(device)

def save_model(modelfile, model, optimizer=None, scheduler=None, epoch=None, mode='parameter'):
    r"""save model to a file

    Parameters
    ----------
    modelfile : str
        model file path
    model : object
        the model object or parameter dict
    optimizer : object or None, optional
        the torch.optim.Optimizer, by default :obj:`None`
    scheduler : object or None, optional
        th.optim.lr_scheduler, by default :obj:`None`
    epoch : int or None, optional
        epoch number, by default :obj:`None`
    mode : str, optional
        saving mode, ``'model'`` means saving model structure and parameters,
        ``'parameter'`` means only saving parameters (default)

    Returns
    -------
    int
        0 is OK
    """

    if type(model) is dict:
        th.save(model, modelfile)
        return 0

    if mode.lower() == 'parameter':
        datadict = {}
        if epoch is not None:
            datadict['epoch'] = epoch
        datadict['network'] = model.state_dict()
        if optimizer is not None:
            datadict['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            datadict['scheduler'] = scheduler.state_dict()
        th.save(datadict, modelfile)
    elif mode.lower() == 'model':
        th.save(model, modelfile)

    return 0

def load_model(modelfile, model=None, optimizer=None, scheduler=None, mode='parameter', device='cpu'):
    r"""load a model from file

    Parameters
    ----------
    modelfile : str
        the model file path
    model : object or None
        the model object or :obj:`None` (default)
    optimizer : object or None, optional
        the torch.optim.Optimizer, by default :obj:`None`
    scheduler : object or None, optional
        th.optim.lr_scheduler, by default :obj:`None`
    mode : str, optional
        the saving mode of model in file, ``'model'`` means saving model structure and parameters,
        ``'parameter'`` means only saving parameters (default)
    device : str, optional
        load model to the specified device
    """

    if model is None:
        model = th.load(modelfile)
        return model.to(device)

    if mode.lower() == 'parameter':
        returns = []
        logdict = th.load(modelfile, map_location=device)
        model.load_state_dict(logdict['network'])
        returns.append(model)
        if optimizer is not None:
            optimizer.load_state_dict(logdict['optimizer'])
            returns.append(optimizer)
        if scheduler is not None:
            scheduler.load_state_dict(logdict['scheduler'])
            returns.append(scheduler)
        return returns[0] if len(returns) == 1 else returns
    elif mode.lower() == 'model':
        model = th.load(modelfile)
        return model.to(device)

def get_parameters(model, optimizer=None, scheduler=None, epoch=None):
    r"""save model to a file

    Parameters
    ----------
    model : object
        the model object
    optimizer : object or None, optional
        the torch.optim.Optimizer, by default :obj:`None`
    scheduler : object or None, optional
        th.optim.lr_scheduler, by default :obj:`None`
    epoch : int or None, optional
        epoch number, by default :obj:`None`

    Returns
    -------
    dict
        keys: 'epoch', 'network' (model.state_dict), 'optimizer' (optimizer.state_dict), 'scheduler' (scheduler.state_dict)
    """

    paramdict = {}
    if epoch is not None:
        paramdict['epoch'] = epoch
    paramdict['network'] = model.state_dict()
    if optimizer is not None:
        paramdict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        paramdict['scheduler'] = scheduler.state_dict()

    return paramdict
