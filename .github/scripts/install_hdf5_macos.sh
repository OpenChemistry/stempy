#!/usr/bin/env bash
set -ev

VERSION="1.14.4"
PATCH_LABEL="3"
VERSION_PATH="hdf5_$VERSION.$PATCH_LABEL"
NAME="hdf5-$VERSION-$PATCH_LABEL"
TARBALL="$NAME.tar.gz"
wget https://github.com/HDFGroup/hdf5/releases/download/$VERSION_PATH/$TARBALL
tar -xzf $TARBALL
mkdir build
cd build
cmake -GNinja ../$NAME -DHDF5_ENABLE_SZIP_ENCODING:BOOL=OFF -DBUILD_TESTING:BOOL=OFF
ninja
sudo ninja install
# We create a symlink to the install directory that doesn't include the version number, so
# we can use it elsewhere without having to change the version number.
sudo ln -s /usr/local/HDF_Group/HDF5/$VERSION /usr/local/HDF_Group/HDF5/current
