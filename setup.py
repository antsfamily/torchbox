#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2130, Zhi Liu.  All rights reserved.

from os import path as os_path
from setuptools import setup
from setuptools import find_packages
from Cython.Build import cythonize
from Cython.Distutils import Extension
from torchbox.version import __version__


this_dir = os_path.abspath(os_path.dirname(__file__))

def read_file(filename):
    with open(os_path.join(filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]

long_description = read_file('README.md'),
long_description_content_type = "text/markdown",

py_extensions = [
              Extension("torchbox.base.arrayops", ['torchbox/base/arrayops.py']), 
              Extension("torchbox.base.baseops", ['torchbox/base/baseops.py']), 
              Extension("torchbox.base.mathops", ['torchbox/base/mathops.py']), 
              Extension("torchbox.base.randomfunc", ['torchbox/base/randomfunc.py']), 
              Extension("torchbox.base.typevalue", ['torchbox/base/typevalue.py']), 
              Extension("torchbox.datasets.mnist", ['torchbox/datasets/mnist.py']), 
              Extension("torchbox.datasets.mstar", ['torchbox/datasets/mstar.py']), 
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
              Extension("torchbox.utils.colormaps", ['torchbox/utils/colormaps.py']),
              Extension("torchbox.utils.colors", ['torchbox/utils/colors.py']),
              Extension("torchbox.utils.convert", ['torchbox/utils/convert.py']),
              Extension("torchbox.utils.file", ['torchbox/utils/file.py']),
              Extension("torchbox.utils.image", ['torchbox/utils/image.py']),
              Extension("torchbox.utils.ios", ['torchbox/utils/ios.py']),
              Extension("torchbox.utils.plot_show", ['torchbox/utils/plot_show.py']),
]

c_extensions = [
              Extension("torchbox.base.arrayops", ['torchbox/base/arrayops.c']), 
              Extension("torchbox.base.baseops", ['torchbox/base/baseops.c']), 
              Extension("torchbox.base.mathops", ['torchbox/base/mathops.c']), 
              Extension("torchbox.base.randomfunc", ['torchbox/base/randomfunc.c']), 
              Extension("torchbox.base.typevalue", ['torchbox/base/typevalue.c']), 
              Extension("torchbox.datasets.mnist", ['torchbox/datasets/mnist.c']), 
              Extension("torchbox.datasets.mstar", ['torchbox/datasets/mstar.c']), 
              Extension("torchbox.misc.draw_shapes", ['torchbox/misc/draw_shapes.c']), 
              Extension("torchbox.misc.noising", ['torchbox/misc/noising.c']),
              Extension("torchbox.misc.sampling", ['torchbox/misc/sampling.c']),
              Extension("torchbox.misc.transform", ['torchbox/misc/transform.c']),
              Extension("torchbox.misc.mapping_operation", ['torchbox/misc/mapping_operation.c']),
              Extension("torchbox.dsp.ffts", ['torchbox/dsp/ffts.c']),
              Extension("torchbox.dsp.convolution", ['torchbox/dsp/convolution.c']),
              Extension("torchbox.dsp.correlation", ['torchbox/dsp/correlation.c']),
              Extension("torchbox.dsp.function_base", ['torchbox/dsp/function_base.c']),
              Extension("torchbox.dsp.window_function", ['torchbox/dsp/window_function.c']),
              Extension("torchbox.dsp.normalsignals", ['torchbox/dsp/normalsignals.c']),
              Extension("torchbox.dsp.polynomialfit", ['torchbox/dsp/polynomialfit.c']),
              Extension("torchbox.linalg.orthogonalization", ['torchbox/linalg/orthogonalization.c']),
              Extension("torchbox.evaluation.contrast", ['torchbox/evaluation/contrast.c']),
              Extension("torchbox.evaluation.entropy", ['torchbox/evaluation/entropy.c']),
              Extension("torchbox.evaluation.error", ['torchbox/evaluation/error.c']),
              Extension("torchbox.evaluation.norm", ['torchbox/evaluation/norm.c']),
              Extension("torchbox.evaluation.snrs", ['torchbox/evaluation/snrs.c']),
              Extension("torchbox.utils.colormaps", ['torchbox/utils/colormaps.c']),
              Extension("torchbox.utils.colors", ['torchbox/utils/colors.c']),
              Extension("torchbox.utils.convert", ['torchbox/utils/convert.c']),
              Extension("torchbox.utils.file", ['torchbox/utils/file.c']),
              Extension("torchbox.utils.image", ['torchbox/utils/image.c']),
              Extension("torchbox.utils.ios", ['torchbox/utils/ios.c']),
              Extension("torchbox.utils.plot_show", ['torchbox/utils/plot_show.c']),
]


try:
    # cythonize(py_extensions)
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
        ext_modules=cythonize(c_extensions)
    )
except:
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
    )
