FROM ubuntu:18.04

# Install deps
RUN apt-get update && \
  apt-get install -y \
  libeigen3-dev \
  libssl1.0-dev \
  git \
  autoconf \
  automake \
  gcc \
  g++ \
  make \
  gfortran \
  wget \
  zlib1g-dev \
  libffi-dev \
  apt-transport-https \
  ca-certificates \
  gnupg \
  software-properties-common && \
  apt-get clean all

# Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - && \
  apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
  apt-get update && \
  apt-get install -y kitware-archive-keyring && \
  apt-key --keyring /etc/apt/trusted.gpg del C1F34CDD40CD72DA && \
  apt-get install -y cmake && \
  apt-get clean all

RUN mkdir /build/ && mkdir /source/

# Build Python
RUN cd /build && wget https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tgz && \
  tar xvzf Python-3.7.3.tgz && cd /build/Python-3.7.3 && \
  ./configure && make -j4 && make install && make clean && rm /build/Python-3.7.3.tgz

# Build mpich
RUN cd /build && wget https://www.mpich.org/static/downloads/3.3/mpich-3.3.tar.gz && \
  tar xvzf mpich-3.3.tar.gz && cd /build/mpich-3.3 && \
  ./configure && make -j4 && make install && make clean && rm /build/mpich-3.3.tar.gz

# Install mpi4py
RUN cd /build && wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.1.tar.gz && \
  tar xvzf mpi4py-3.0.1.tar.gz

RUN cd /build/mpi4py-3.0.1 && python3 setup.py build && python3 setup.py install

# Build VTK-m
RUN cd /source && \
  wget https://gitlab.kitware.com/vtk/vtk-m/-/archive/master/vtk-m-master.tar.gz && \
  tar zxvf vtk-m-master.tar.gz && \
  rm vtk-m-master.tar.gz

RUN mkdir -p /build/vtk-m && \
  cd /build/vtk-m && \
  cmake -DCMAKE_BUILD_TYPE:STRING=Release \
  -DBUILD_SHARED_LIBS:BOOL=OFF \
  -DVTKm_ENABLE_OPENMP:BOOL=ON \
  -DVTKm_ENABLE_RENDERING:BOOL=OFF \
  -DVTKm_ENABLE_TESTING:BOOL=OFF \
  /source/vtk-m-master . && \
  make -j4

# Build stempy
COPY . /source/stempy

RUN mkdir -p /build/stempy && \
  cd /build/stempy && \
  cmake -DCMAKE_BUILD_TYPE:STRING=Release \
  -Dstempy_ENABLE_VTKm:BOOL=ON \
  -DVTKm_DIR:PATH=/build/vtk-m/lib/cmake/vtkm-1.3 \
  /source/stempy . && \
  make -j4

# Install stempy
RUN pip3 install /source/stempy && \
  cp /build/stempy/lib/stempy/*.so /usr/local/lib/python3.7/site-packages/stempy/ && \
  pip3 install ipykernel matplotlib

RUN rm -rf /build

RUN /sbin/ldconfig
