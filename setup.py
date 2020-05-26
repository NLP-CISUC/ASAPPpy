#! /usr/bin/env python
#
# Copyright (c) 2020 José Santos
# License: MIT License
from setuptools import setup, find_packages

import ASAPPpy

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='ASAPPpy',
    version=ASAPPpy.__version__,
    description='Semantic Textual Similarity and Dialogue System package for Python',
    long_description=LONG_DESCRIPTION,

    packages=find_packages(),

    author=u'José Santos',
    author_email='santos@student.dei.uc.pt',

    license='MIT License',

    keywords='Natural Language Processing, NLP,'
        'Sentence Similarity, Semantic Textual Similarity, STS,'
        'Dialogue Agents, Chatbot Framework, Chatbot',

    platforms='any',

    zip_safe=False,

    classifiers=[  # from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],

    python_requires='>=3.6.1',
)