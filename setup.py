#!/usr/bin/env python

from setuptools import (
        setup,
        find_packages,
        )

VERSION = '0.0.1'

setup(
        name='torch_igt',
        packages=find_packages(),
        version=VERSION,
        description='PyTorch implementation of Accelerated Implicit Gradient Transport.',
        author='Anonymous',
        author_email='ano@nymous.com',
        url = 'https://github.com/',
        license='License :: OSI Approved :: Apache Software License',
        classifiers=[],
        scripts=[],
)
