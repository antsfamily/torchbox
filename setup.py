#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2030, Zhi Liu.  All rights reserved.

from os import path as os_path
from setuptools import setup
from setuptools import find_packages
from Cython.Build import cythonize
from Cython.Distutils import Extension
from torchbox.version import __version__


this_dir = os_path.abspath(os_path.dirname(__file__))

extensions = [
              Extension("torchbox.base.arrayops", ['torchbox/base/arrayops.py']), 
              Extension("torchbox.base.baseops", ['torchbox/base/baseops.py']), 
              Extension("torchbox.base.mathops", ['torchbox/base/mathops.py']), 
              Extension("torchbox.base.randomfunc", ['torchbox/base/randomfunc.py']), 
              Extension("torchbox.misc.draw_shapes", ['torchbox/misc/draw_shapes.py']), 
              Extension("torchbox.misc.noising", ['torchbox/misc/noising.py']),
              Extension("torchbox.misc.sampling", ['torchbox/misc/sampling.py']),
              Extension("torchbox.misc.transform", ['torchbox/misc/transform.py']),
              Extension("torchbox.misc.mapping_operation", ['torchbox/misc/mapping_operation.py']),
              Extension("torchbox.dsp.ffts", ['torchbox/dsp/ffts.py']),
              Extension("torchbox.dsp.convolution", ['torchbox/dsp/convolution.py']),
              Extension("torchbox.dsp.correlation", ['torchbox/dsp/correlation.py']),
              Extension("torchbox.dsp.function_base", ['torchbox/dsp/function_base.py']),
              Extension("torchbox.dsp.window_function", ['torchbox/dsp/window_function.py']),
              Extension("torchbox.dsp.normalsignals", ['torchbox/dsp/normalsignals.py']),
              Extension("torchbox.dsp.polynomialfit", ['torchbox/dsp/polynomialfit.py']),
              Extension("torchbox.linalg.orthogonalization", ['torchbox/linalg/orthogonalization.py']),
              Extension("torchbox.evaluation.contrast", ['torchbox/evaluation/contrast.py']),
              Extension("torchbox.evaluation.entropy", ['torchbox/evaluation/entropy.py']),
              Extension("torchbox.evaluation.error", ['torchbox/evaluation/error.py']),
              Extension("torchbox.evaluation.norm", ['torchbox/evaluation/norm.py']),
              Extension("torchbox.evaluation.snrs", ['torchbox/evaluation/snrs.py']),
]

def read_file(filename):
    with open(os_path.join(filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]

long_description = read_file('README.md'),
long_description_content_type = "text/markdown",

setup(name='torchbox',
      version=__version__,
      description="A PyTorch Toolbox.",
      author='Zhi Liu',
      author_email='zhiliu.mind@gmail.com',
      url='https://iridescent.ink/torchbox/',
      download_url='https://github.com/antsfamily/torchbox/',
      license='MIT',
      packages=find_packages(),
      install_requires=read_requirements('requirements.txt'),
      include_package_data=True,
      keywords=['PyTorch', 'Machine Learning', 'Signal Processing', 'Deep Learning'],
      ext_modules=cythonize(extensions)
)


