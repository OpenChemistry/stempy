import os
from setuptools import setup, find_packages

setup(
    name='stempy',
    version='0.0.1',
    setup_requires=['setuptools_scm'],
    description='A package for the ingestion of 4D STEM data.',
    long_description='A package for the ingestion of 4D STEM data.',
    url='https://github.com/OpenChemistry/stempy',
    author='Kitware Inc',
    license='BSD 3-Clause',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
    ],
    keywords='',
    packages=find_packages(),
    install_requires=[]
)
