Building Stempy
===============

The following system dependencies are required for building stempy:

 * CMake 3.5+
 * Python 3.6+
 * Git 2.1+
 * C++ compiler with C++14 support
 * Eigen 3.3.0+

Virtual Environment (optional)
------------------------------
When building stempy, we typically use a [virtualenv](https://virtualenv.pypa.io/en/latest/)
to create an isolated environment. `virtualenv` can be installed via `pip3 install virtualenv`.

An appropriate virtual environment may be created like so `virtualenv -p python3.6 stempy`. When
building and running stempy, ensure that this environment is active.

Building VTK-m (optional)
-------------------------
[VTK-m](https://gitlab.kitware.com/vtk/vtk-m) is an optional dependency that allows many
of the algorithms to run in parallel. It may be built with commands similar to the
following (this example uses the [ninja build system](https://ninja-build.org/)):
```
git clone https://gitlab.kitware.com/vtk/vtk-m
mkdir vtkm-build
cd vtkm-build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DVTKm_ENABLE_OPENMP=ON \
  -DVTKm_ENABLE_CUDA=OFF \
  -DVTKm_ENABLE_TBB=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DVTKm_ENABLE_TESTING=OFF \
  -DVTKm_ENABLE_RENDERING=OFF \
  -DCMAKE_INSTALL_PREFIX=./install \
  ../vtk-m -G Ninja
ninja install
 ````
Building in release mode is particularly important for doing benchmarks.

This VTK-m build will allow VTK-m to run with OpenMP. Other types of parallelism
such as CUDA or TBB may be used by enabling their respective options. However,
note that if VTK-m is built with CUDA turned on, stempy *must* also be
built with CUDA turned on, or there will be compile errors.

Building Stempy
---------------
Stempy may be built using instructions similar to the following
(this example uses the [ninja build system](https://ninja-build.org/)):
```
git clone --recursive https://github.com/openchemistry/stempy
pip install -e stempy
mkdir stempy-build
cd stempy-build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -Dstempy_ENABLE_VTKm=ON \
  -Dstempy_ENABLE_CUDA=OFF \
  -DVTKm_DIR=$PWD/../vtkm-build/install/lib/cmake/vtkm-1.4 \
  ../../src/stempy -G Ninja
ninja
```

This builds stempy with VTK-m enabled, using OpenMP by default. If stempy is to be
built without VTKm, just leave out all of the VTKm options.

Once it has finished building, a soft link will need to be created in the python
`site-packages` directory that refers to the `lib/stempy` directory in the stempy
build directory. An example (ran from the build directory) can be shown below:
```
ln -s $PWD/lib/stempy $HOME/virtualenvs/stempy/lib/python3.6/site-packages/
```

Once this has been completed, stempy should be officially ready for use!
