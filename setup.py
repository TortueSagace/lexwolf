#!/usr/bin/env python3

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="lexwolf",
    version="0.2.3",
    description="The LexWolf chess artificial intelligences.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Alexandre Le Mercier",
    author_email="alexandre.le.mercier@ulb.be",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=["lexwolf"],
    include_package_data=True,
    install_requires=["chess"]
)

"""
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
"""