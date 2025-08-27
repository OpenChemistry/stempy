#!/usr/bin/env bash

# arch=x86_64
arch=aarch64

if [ "$arch" = "x86_64" ]; then
    platform=linux/amd64
elif [ "$arch" = "aarch64" ]; then
    platform=linux/arm64
fi

image_base=quay.io/pypa/manylinux_2_28_${arch}
tag=openchemistry/stempy_wheel_builder_${arch}

docker build . --platform=$platform -t $tag --build-arg BASE_IMAGE=$image_base --build-arg ARCH=$arch
docker push $tag
