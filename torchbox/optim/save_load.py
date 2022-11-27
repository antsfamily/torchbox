#!/usr/bin/env python
#-*- coding: utf-8 -*-
# @file      : save_load.py
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


def device_transfer(obj, name, device):
    if name in ['optimizer', 'Optimizer']:
        for group in obj.param_groups:
            for p in group['params']:
                for k, v in obj.state[p].items():
                    if th.is_tensor(v):
                        obj.state[p][k] = v.to(device)
  
                    


