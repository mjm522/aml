#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup()
d['packages'] = ['rl_algos']
d['package_dir'] = {'': 'src'}

setup(**d)
