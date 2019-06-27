import os
from setuptools import setup

setup(
    name='stempy',
    use_scm_version=True,
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
    packages=['stempy', 'stempy.io', 'stempy.image'],
    package_dir={'':'python'},
    install_requires=[
        'numpy',
        'h5py',
        'click'
    ]
)
