#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-23 07:01:55
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$


from __future__ import division, print_function, absolute_import
import re


def str2list(s):
    """Converts string with ``[`` and ``]`` to list

    Parameters
    ----------
    s : str
        The string.
    """
    left = [i.start() for i in re.finditer(r'\[', s)]
    print(left)
    right = [i.start() for i in re.finditer(r'\]', s)]
    print(right)

    nlevel = -1
    for l in left:
        nlevel += 1
        if l > right[0]:
            break
    right[0:nlevel - 1] = right[0:nlevel - 1][::-1]
    right.insert(0, right.pop())
    print(right)


def str2num(s, tfunc=None):
    """Extracts numbers in a string.

    Parameters
    ----------
    s : str
        The string.
    tfunc : None, optional
        formating function.

    Returns
    -------
    list
        The number list.
    """
    numstr = re.findall(r'-?\d+\.?\d*e*E?[-+]?\d*', s)
    if tfunc is None:
        return numstr
    else:
        if tfunc == 'auto':
            numlist = []
            for num in numstr:
                if num.find('.') > -1 or num.find('e') > -1:
                    numlist.append(float(num))
                else:
                    numlist.append(int(num))
            return numlist
        else:
            return [tfunc(i) for i in numstr]


def str2sec(x, sep=':'):
    """Extracts second in a time string.
        
        ``hh:mm:ss``  -->  ``hh*3600 + mm*60 + ss``

    Parameters
    ----------
    s : str
        The string or string list/tuple.
    sep : str
        The separator between hour, minute and seconds, default is ``':'``.

    Returns
    -------
    y : int
        The seconds.
    """
    if type(x) is str:
        h, m, s = x.strip().split(sep)
        return int(h) * 3600 + int(m) * 60 + int(s)
    
    if (type(x) is list) or (type(x) is tuple):
        y = []
        for xi in x:
            h, m, s = xi.strip().split(sep)
            y.append(int(h) * 3600 + int(m) * 60 + int(s))
        return y


if __name__ == '__main__':

    s = '[0, [[[[1], 2.], 33], 4], [5, [6, 2.E-3]], 7, [8]], 1e-3'

    str2list(s)

    # print(str2num(s, int))
    print(str2num(s, float))
    print(str2num(s, 'auto'))

    print(2**(str2num('int8', int)[0]))
    print(str2num('int', int) == [])

    print(str2sec('1:00:0'))
    print(str2sec('1:10:0'))
    print(str2sec('1:10:6'))
    print(str2sec('1:10:30'))
