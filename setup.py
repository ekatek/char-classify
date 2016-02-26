
"""A setuptools based setup module."""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    # Project basic metadata
    name='chClassifier',
    version='0.6.2',
    description='Neural Network to classify short strings',
    long_description=long_description,
    url='https://github.com/ekatek/char-classify',
    license='MIT',

    # Author details
    author='Ekate Kuznetsova',
    author_email='char-classify@technekate.com',

    # Project extended metadata
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Text Processing',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='deep-learning machine-learning string classifier',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),


    # List run-time dependencies here.
    install_requires= ['chainer', 'numpy'],
)


