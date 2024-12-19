#!/usr/bin/env bash

# arch=x86_64
arch=aarch64
image_base=quay.io/pypa/manylinux_2_28_${arch}
tag=samwelborn/stempy_wheel_builder_${arch}

docker build . --platform=linux/amd64,linux/arm64 -t $tag --build-arg BASE_IMAGE=$image_base --build-arg ARCH=$arch
docker push $tag
