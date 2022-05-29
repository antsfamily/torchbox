#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


from __future__ import division, print_function, absolute_import
import os
import re
from torchbox.utils.convert import str2sec


def listxfile(listdir=None, exts=None, recursive=False, filelist=[]):
    r"""List the files in a directory.


    Parameters
    ----------
    listdir : None, optional
        The directory for listing. The default is None, which means return :attr:`filelist` directly.
    exts : str, list or None, optional
        File extension string or extension string list, such as ``'.png'`` or ``['.png', 'jpg']``.
        The default is None, which means any extension.
    recursive : bool, optional
        Recursive search? The default is False, which means only list the root directory (:attr:`listdir`).
    filelist : list, optional
        An initial list contains files's path. The default is ``[]``.

    Returns
    -------
    list
        The list of file path with extension :attr:`exts`.

    """

    if listdir is None:
        return filelist

    exts = [exts] if type(exts) is str else exts

    for s in os.listdir(listdir):
        newDir = os.path.join(listdir, s)
        if os.path.isfile(newDir):
            if exts is not None:
                if newDir and(os.path.splitext(newDir)[1] in exts):
                    filelist.append(newDir)
            else:
                filelist.append(newDir)
        else:
            if recursive:
                listxfile(listdir=newDir, exts=exts, recursive=True, filelist=filelist)

    return filelist


def pathjoin(*kwargs):
    """Joint strings to a path.

    Parameters
    ----------
    *kwargs
        strings.

    Returns
    -------
    str
        The joined path string.
    """

    filesep = os.path.sep
    filepath = ''
    for k in kwargs:
        filepath += filesep + k
    return filepath[1:]


def fileparts(file):
    r"""Filename parts

    Returns the path, file name, and file name extension for the specified :attr:`file`.
    The :attr:`file` input is the name of a file or folder, and can include a path and
    file name extension.

    Parameters
    ----------
    file : str
        The name string of a file or folder.

    Returns
    -------
    filepath : str
        The path string of the file or folder.
    name : str
        The name string of the file or folder.
    ext : str
        The extension string of the file.

    """

    filepath, filename = os.path.split(file)
    name, ext = os.path.splitext(filename)
    return filepath, name, ext


def readtxt(filepath, mode=None):
    """Read a text file.

    Parameters
    ----------
    filepath : str
        The path string of the file.
    mode : str or None, optional
        ``'line'`` --> reading line-by-line.
        ``'all'`` --> reading all content.

    Returns
    -------
    str
        File content.
    """
    if mode is None:
        mode = 'all'

    with open(filepath, "r") as f:
        if mode == 'all':
            data = f.read()
        if mode == 'line':
            data = f.readlines()
            N = len(data)
            for n in range(N):
                data[n] = data[n].strip('\n')
        f.close()
    return data


def readnum(filepath, pmain='Train', psub='loss', vfn=float, nshots=None):
    """Read a file and extract numbers in it.

    Parameters
    ----------
    filepath : str
        Description
    pmain : str, optional
        The matching pattern string, such as '--->Train'.
    psub : str, optional
        The sub-matching pattern string, such as 'loss'.
    vfn : function, optional
        The function for formating the numbers. ``float`` --> convert to float number; ``int`` --> convert to integer number..., The default is ``float``.
    nshots : None, optional
        The number of shots of sub-matching pattern.

    Returns
    -------
    str
        The list of numbers.
    """
    if nshots is None:
        nshots = float('Inf')
    v = []
    cnt = 1
    with open(filepath, 'r') as f:
        while True:
            datastr = f.readline()
            posmain = datastr.find(pmain)
            if datastr == '':
                break
            if posmain > -1:
                # print(datastr, posmain)
                aa = re.findall(psub + r'-?\d+\.?\d*e*E?[-+]?\d*', datastr)
                # aa = re.findall(psub + r'\d\.?\d*', datastr))
                if aa != []:
                    v.append(vfn(aa[0][len(psub):]))
                    cnt += 1
            if cnt > nshots:
                break
    return v


def readcsv(filepath, sep=None, vfn=None, nlines=None):
    """Read a csv file and extract numbers in it.

    Parameters
    ----------
    filepath : str
        The path string of the file.
    sep : str, optional
        The separation character. Such as ``','`` or ``' '``. If None (default) or ``''`` (empty) return a list of all the lines.
    vfn : function or None, optional
        The function for formating the numbers. ``float`` --> convert to float number; ``int`` --> convert to integer number..., The default is :obj:`None`, which means won't converted, string format.
    nlines : None, optional
        The number of lines for reading, the default is None, which means all the lines.

    Returns
    -------
    list
        The list of numbers or strings.
    """
    if nlines is None:
        nlines = float('Inf')
    numbers = []
    cnt = 1
    with open(filepath, 'r') as f:
        while True:
            datastr = f.readline().strip()
            if not datastr:
                break
            if datastr != '':
                if (sep is not None) and sep != '':
                    datalist = datastr.split(sep)
                    if vfn is None:
                        numbers.append(datalist)
                    else:
                        numbers.append([vfn(v) for v in datalist])
                else:
                    numbers.append(datastr)
                cnt += 1
            if cnt > nlines:
                break
    return numbers


def readsec(filepath, pmain='Train', psub='time: ', vfn=int, nshots=None):
    """Read a file and extract seconds in it.

        ``hh:mm:ss``  -->  ``hh*3600 + mm*60 + ss``

    Parameters
    ----------
    filepath : str
        The path string of the file.
    pmain : str, optional
        The matching pattern string, such as '--->Train'.
    psub : str, optional
        The sub-matching pattern string, such as 'loss'.
    vfn : function or None, optional
        The function for formating the numbers. ``float`` --> convert to float number; ``int`` --> convert to integer number.
    nshots : None, optional
        The number of shots of sub-matching pattern.

    Returns
    -------
    list
        The list of seconds.
    """
    if nshots is None:
        nshots = float('Inf')
    v = []
    cnt = 1
    with open(filepath, 'r') as f:
        while True:
            datastr = f.readline()
            posmain = datastr.find(pmain)
            if datastr == '':
                break
            if posmain > -1:
                aa = re.findall(psub + r'\d+:\d+:\d+', datastr)
                if aa != []:
                    v.append(vfn(str2sec(aa[0][len(psub):])))
                    cnt += 1
            if cnt > nshots:
                break
    return v


if __name__ == '__main__':

    files = listxfile(listdir='/home/liu/', exts=None, recursive=False, filelist=[])
    print(files)

    filepath = pathjoin('a', 'b', 'c', '.d')
    print(filepath)

    filepath = '../../data/files/log.log'

    v = readnum(filepath, pmain='Train', psub='loss: ', vfn=float, nshots=10)
    print(v)
