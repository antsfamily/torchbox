#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from __future__ import division, print_function, absolute_import
import h5py
import json
import yaml
import numpy as np
import scipy.io as scio
from torchbox.base.baseops import dreplace


def loadyaml(filepath, field=None):
    """Load a yaml file.

    Parameters
    ----------
    filepath : str
        The file path string.
    field : None, optional
        The string of field that want to be loaded.

    """
    f = open(filepath, 'r', encoding='utf-8')
    if field is None:
        if int(yaml.__version__[0]) > 3:
            data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            data = yaml.load(f)
    else:
        if int(yaml.__version__[0]) > 3:
            data = yaml.load(f, Loader=yaml.FullLoader)[field]
        else:
            data = yaml.load(f)
    return data


def loadjson(filepath, field=None):
    """load a json file

    Parameters
    ----------
    filepath : str
        The file path string.
    field : None, optional
        The string of field that want to be loaded.

    """
    with open(filepath, 'r', encoding='utf-8') as f:
        if field is None:
            data = json.load(f)
        else:
            data = json.load(f)[field]
    return data


def _check_keys(d):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in d:
        if isinstance(d[key], scio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
    return d


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    d = {}
    # print(dir(matobj),  "jjjj")
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scio.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        else:
            d[strg] = elem
    return d


def loadmat(filepath):
    """load data from an ``.mat`` file

    load data from an ``.mat`` file (``'None'`` will be replaced by :obj:`None`)

    see https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

    Parameters
    ----------
    filepath : str
        The file path string.

    """
    mdict = scio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    mdict = _check_keys(mdict)
    dreplace(mdict, fv='None', rv=None, new=False)
    del mdict['__header__'], mdict['__version__'], mdict['__globals__']

    return mdict


def savemat(filepath, mdict, fmt='5'):
    """save data to an ``.mat`` file

    save data to ``.mat`` file (:obj:`None` will be replaced by ``'None'``)

    Parameters
    ----------
    filepath : str
        savefile path
    mdict : dict
        data in dict formation. 
    fmt : str, optional
        mat formation, by default '5'

    Returns
    -------
    0
        all is ok!
    """
    dreplace(mdict, fv=None, rv='None', new=False)
    scio.savemat(filepath, mdict, format=fmt)
    dreplace(mdict, fv='None', rv=None, new=False)

    return 0


def _create_group_dataset(group, mdict):
    for k, v in mdict.items():
        if k in group.keys():
            del group[k]

        if type(v) is dict:
            subgroup = group.create_group(k)
            _create_group_dataset(subgroup, v)
        else:
            group.create_dataset(k, data=v)


def _read_group_dataset(group, mdict, keys=None):
    if keys is None:
        for k in group.keys():
            if type(group[k]) is h5py.Group:
                exec(k + '={}')
                _read_group_dataset(group[k], eval(k))
                mdict[k] = eval(k)
            else:
                mdict[k] = group[k][()]
    else:
        for k in keys:
            if type(group[k]) is h5py.Group:
                exec(k + '={}')
                _read_group_dataset(group[k], eval(k))
                mdict[k] = eval(k)
            else:
                mdict[k] = group[k][()]


def loadh5(filename, keys=None):
    """load h5 file

    load data from a h5 file. (``'None'`` will be replaced by :obj:`None`)

    Parameters
    ----------
    filename : str
        File's full path string.
    keys : list
        list of keys.

    Returns
    -------
    D : dict
        The loaded data in ``dict`` type.

    """

    f = h5py.File(filename, 'r')
    D = {}

    _read_group_dataset(f, D, keys)

    dreplace(D, fv='None', rv=None, new=False)

    f.close()
    return D


def saveh5(filename, mdict, mode='w'):
    """save data to h5 file

    save data to h5 file (:obj:`None` will be replaced by ``'None'``)

    Parameters
    ----------
    filename : str
        filename string
    mdict : dict
        each dict is store in group, the elements in dict are store in dataset
    mode : str
        save mode, ``'w'`` for write, ``'a'`` for add.

    Returns
    -------
    number
        0 --> all is well.
    """

    dreplace(mdict, fv=None, rv='None', new=False)
    f = h5py.File(filename, mode)

    _create_group_dataset(f, mdict)

    f.close()

    dreplace(mdict, fv='None', rv=None, new=False)

    return 0


def mvkeyh5(filepath, ksf, kst, sep='.'):
    """rename keys in ``.h5`` file

    Parameters
    ----------
    filepath : str
        The file path string
    ksf : list
        keys from list, e.g. ['a.x', 'b.y']
    kst : list
        keys to list, e.g. ['a.1', 'b.2']
    sep : str, optional
        The separate pattern, default is ``'.'``

    Returns
    -------
    0
        All is ok!
    """
    ksf = [ksf] if type(ksf) is not list else ksf
    kst = [kst] if type(kst) is not list else kst
    f = h5py.File(filepath, 'a')
    for keyf, keyt in zip(ksf, kst):
        keyf = keyf.split(sep)
        keyt = keyt.split(sep)
        grp = f
        for kf, kt in zip(keyf[:-1], keyt[:-1]):
            grp = grp[kf]
        grp.create_dataset(keyt[-1], data=grp[keyf[-1]][()])
        del grp[keyf[-1]]
    f.close()
    return 0


if __name__ == '__main__':

    a = np.random.randn(3, 4)
    b = 10
    c = [1, 2, 3]
    d = {'d1': 1, 'd2': a}
    s = 'Hello, the future!'
    t = (0, 1)
    n = None

    savemat('./data.mat', {'a': {'x': a, 'y': 1}, 'b': b, 'c': c, 'd': d, 's': s, 'n': n})
    saveh5('./data.h5', {'a': {'x': a}, 'b': b, 'c': c, 'd': d, 's': s, 'n': n})
    data = loadh5('./data.h5', keys=['a', 'd', 's', 'n'])
    for k, v in data.items():
        print(k, v)
    print(data.keys())
    print("==========1")

    data = loadmat('./data.mat')
    for k, v in data.items():
        print(k, v)

    print("==========2")
    # saveh5('./data.h5', {'t': t}, 'w')
    saveh5('./data.h5', {'t': t}, 'a')
    saveh5('./data.h5', {'t': (2, 3, 4)}, 'a')
    data = loadh5('./data.h5')

    for k, v in data.items():
        print(k, v)

    mvkeyh5('./data.h5', ['a.x'], ['a.1'])
    data = loadh5('./data.h5')

    for k, v in data.items():
        print(k, v)
