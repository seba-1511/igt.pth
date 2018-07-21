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
        author='Seb Arnold',
        author_email='smr.arnold@gmail.com',
        url = 'https://github.com/seba-1511/igt.pth',
        download_url = 'https://github.com/seba-1511/igt.pth/archive/' + str(VERSION) + '.zip',
        license='License :: OSI Approved :: Apache Software License',
        classifiers=[],
        scripts=[],
)
