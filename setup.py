import os

from skbuild import setup


def extra_cmake_args():
    # FIXME: this doesn't seem to work if we supply more than one argument.
    # I really am not sure why.
    env = os.getenv('EXTRA_CMAKE_ARGS')
    return env.split(';') if env else []


cmake_args = [] + extra_cmake_args()

setup(
    name='stempy',
    use_scm_version=True,
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
    packages=['stempy'],
    install_requires=[
        'numpy',
        'h5py',
        'deprecation'
    ],
    cmake_args=cmake_args,
)
