#! /usr/bin/env python
#
# Copyright (c) 2020 José Santos
# License: MIT License
from setuptools import setup, find_packages
from setuptools.command.install import install

import ASAPPpy

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='ASAPPpy',
    version=ASAPPpy.__version__,
    description='Semantic Textual Similarity and Dialogue System package for Python',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    
    packages=find_packages(),
    include_package_data=True,

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
    install_requires=[
        'setuptools == 49.6.0',
        'scikit-learn == 0.22.2',
        'pandas >= 1.1.1',
        'requests',
        'slackclient == 2.1.0',
        'slackeventsapi == 2.1.0',
        'nltk == 3.2.5',
        'NLPyPort == 2.2.5',
        'spacy',
        'gensim',
        'joblib',
        'num2words',
        'Whoosh',
        'Keras',
        'tensorflow',
        'cufflinks',
        'matplotlib',
        'seaborn',
        'imblearn',
    ],
)