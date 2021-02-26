ARG BASE_IMAGE=quay.io/pypa/manylinux2010_x86_64
FROM ${BASE_IMAGE}

RUN yum install -y \
  git \
  wget \
  eigen3-devel \
  hdf5-devel

# Install the latest cmake
RUN VERSION=3.19.4 && \
  wget -q https://github.com/Kitware/CMake/releases/download/v$VERSION/cmake-$VERSION-Linux-x86_64.sh && \
  bash cmake-$VERSION-Linux-x86_64.sh --skip-license --prefix=/usr/local && \
  rm cmake-$VERSION-Linux-x86_64.sh

# Apply the code patch from https://github.com/pypa/auditwheel/pull/283
# This allows for non-python dependencies to get their RPATHs patched, which
# we need so that libstem.so can get patched for HDF5
# FIXME: when https://github.com/pypa/auditwheel/pull/283 is merged and
# a part of quay.io/pypa/manylinux2010_x86_64, we can remove these lines.
COPY auditwheel_pr_283.patch /
RUN cd /opt/_internal/tools/lib/python3.7/site-packages && \
  patch -p1 < /auditwheel_pr_283.patch && \
  rm /auditwheel_pr_283.patch
