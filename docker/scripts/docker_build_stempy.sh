#!/bin/sh

GIT_TAG=`git log -1 --pretty=%h`
DOCKERHUB_ORG="openchemistry"
PYTHON_VERSION="3.9"
PYTHON_VERSION_NODOT=$(echo $PYTHON_VERSION | tr -d .)
BASE_IMAGE=${DOCKERHUB_ORG}/stempy:py${PYTHON_VERSION_NODOT}-base${DEV}
STEMPY_TAG=${DOCKERHUB_ORG}/stempy${IPYKERNEL}:py${PYTHON_VERSION_NODOT}-${GIT_TAG}
RELEASE_OR_DEBUG="Release" # can change to "RelWithDebInfo" or "Debug"
DEV="" # can change to "-dev" if gdb/vim are desired inside container
IPYKERNEL="" # can change to "ipykernel" if building ipykernel version

docker build \
--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
--build-arg DEV=${DEV} \
--load \
-t ${BASE_IMAGE} \
--target base \
-f ../Dockerfile.base \
../../ $@

docker build \
--build-arg BASE_IMAGE=${BASE_IMAGE} \
--build-arg MPI=OFF \
--build-arg RELEASE_OR_DEBUG=${RELEASE_OR_DEBUG} \
--build-arg IPYKERNEL="" \
--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
--load \
-t ${STEMPY_TAG} \
-f ../Dockerfile.stempy \
../../ $@