#!/usr/bin/env bash
set -ev

choco install ninja
git clone --recursive -b 3.3.9 --depth 1 https://gitlab.com/libeigen/eigen /c/eigen
mkdir /c/build
cd /c/build
cmake -GNinja /c/eigen
ninja install
