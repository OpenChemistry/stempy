# Building stempy Shifter Image

1. Build the docker image:
    `cd <stempy_repo>`  
    `docker build -t openchemistry/stempy -f docker/Dockerfile .`
2. Push the image to dockerhub:
    `docker push openchemistry/stempy`
3. Pull the new image into shifter ( need to be run on cori ):
    shifterimg pull openchemistry/stempy
   N.B. I have noticed that it takes sometime for shifter to realize that a new image is available, so sometimes you have to wait say 10 minutes after pushing to dockerhub, not sure why!

# GitHub Actions

The GitHub Actions docker workflow (.github/workflows/docker.yml) will automatically build the images in docker/stempy-conda-jammy, docker/stempy-mpi-conda, and stempy-mpi-conda-jammy. These images run with conda's python. The base image (`Docker.base`) will only be built if "trigger-ci" is in the head commit or if it changes in the commit. The workflow will check on these things, and then proceed to building the images for stempy. This significantly improves build time - we shouldn't have to build mpich every time. 

The original versions are also built with GHA in a separate job in the same docker GHA workflow.