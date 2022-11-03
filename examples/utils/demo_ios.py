#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
import torchbox as tb

a = np.random.randn(3, 4)
b = 10
c = [1, 2, 3]
d = {'1': 1, '2': a}
s = 'Hello, the future!'
t = (0, 1)

# tb.savemat('./data.mat', {'a': a, 'b': b, 'c': c, 'd': d, 's': s})
# data = tb.loadmat('./data.mat')
# print(data.keys())

# print("==========")
# tb.saveh5('./data.h5', {'a': a, 'b': b, 'c': c, 'd': d, 's': s})
# data = tb.loadh5('./data.h5', keys=['a', 's'])
# print(data.keys())

# print("==========")
# # saveh5('./data.h5', {'t': t}, 'w')
# tb.saveh5('./data.h5', {'t': t}, 'a')
# tb.saveh5('./data.h5', {'t': (2, 3, 4)}, 'a')
# data = tb.loadh5('./data.h5')

# for k, v in data.items():
#     print(k, v)


x = tb.loadyaml('data/files/demo.yaml', 'trainid')
print(x, type(x))
x = tb.loadjson('data/files/demo.json', 'trainid')
print(x, type(x))

x = tb.loadyaml('data/files/demo.yaml')
print(x, type(x))
# x = tb.loadjson('data/files/demo.json')
# print(x, type(x))

tb.saveyaml('data/files/demo1.yaml', x, indent='  ', mode='a')
x = tb.loadyaml('data/files/demo1.yaml')
print(x, type(x))


# from ruamel.yaml import YAML
# yaml = YAML()
# with open('data/file/demo1.yaml', 'w') as f:
#     yaml.dump(x, f)
