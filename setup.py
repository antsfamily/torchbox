#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2130, Zhi Liu.  All rights reserved.

import os
from setuptools import setup
from setuptools import find_packages
from Cython.Build import cythonize
from Cython.Distutils import Extension


pkgname = 'torchbox'
this_dir = os.path.abspath(os.path.dirname(__file__))

def read_requirements(filename):
    return [line.strip() for line in open(filename, encoding="utf-8").read().splitlines()
            if (not line.startswith('#') and len(line)>0)]

def __listxfile__(listdir=None, exts=None, recursive=False, filelist=[]):
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
                __listxfile__(listdir=newDir, exts=exts, recursive=True, filelist=filelist)

    return filelist


def listxfile(listdir=None, exts=None, recursive=False):
    return __listxfile__(listdir=listdir, exts=exts, recursive=recursive, filelist=[])

version = open(os.path.join(this_dir, pkgname, 'version.py'), encoding="utf-8").read().strip().split('=')[-1].strip().strip('\'').strip('\"')
requirements = read_requirements(os.path.join(this_dir, 'requirements.txt'))
long_description = open(os.path.join(this_dir, 'README.md'), encoding="utf-8").read()

fext = '.pyi'
modules = listxfile(os.path.join(this_dir, pkgname), fext, recursive=True)
modules = [m[len(this_dir)+1:] for m in modules]
py_extensions, c_extensions = [], []
for efile in modules:
    if efile.find('__init__.py') < 0:
        efile = efile[:-len(fext)]
        py_extensions.append(Extension(efile.replace(os.sep, '.'), [efile + '.py']))
        c_extensions.append(Extension(efile.replace(os.sep, '.'), [efile + '.c']))

try:
    cythonize(py_extensions, language_level=3)
    setup(name=pkgname,
        version=version,
        description="A PyTorch Toolbox.",
        long_description=long_description,
        author='Zhi Liu',
        author_email='zhiliu.mind@gmail.com',
        url='https://iridescent.ink/%s/' % pkgname,
        download_url='https://github.com/antsfamily/%s/' % pkgname,
        license='MIT',
        packages=find_packages(),
        install_requires=requirements,
        include_package_data=True,
        keywords=['PyTorch', 'Machine Learning', 'Signal Processing', 'Deep Learning'],
        ext_modules=cythonize(c_extensions, language_level=3)
    )
except:
    setup(name=pkgname,
        version=version,
        description="A PyTorch Toolbox.",
        long_description=long_description,
        author='Zhi Liu',
        author_email='zhiliu.mind@gmail.com',
        url='https://iridescent.ink/%s/' % pkgname,
        download_url='https://github.com/antsfamily/%s/' % pkgname,
        license='MIT',
        packages=find_packages(),
        install_requires=requirements,
        include_package_data=True,
        keywords=['PyTorch', 'Machine Learning', 'Signal Processing', 'Deep Learning'],
    )
