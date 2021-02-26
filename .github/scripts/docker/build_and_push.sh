#!/usr/bin/env bash

image_base=quay.io/pypa/manylinux2010_x86_64
tag=openchemistry/stempy_wheel_builder

docker build . -t $tag --build-arg BASE_IMAGE=$image_base
docker push $tag
