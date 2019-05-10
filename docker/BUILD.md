# Building stempy Shifter Image

1. Build the docker image:
    `cd <stempy_repo>`  
    `docker build -t openchemistry/stempy -f docker/Dockerfile .`
2. Push the image to dockerhub:
    `docker push openchemistry/stempy`
3. Pull the new image into shifter ( need to be run on cori ):
    shifterimg pull openchemistry/stempy
   N.B. I have noticed that it takes sometime for shifter to realize that a new image is available, so sometimes you have to wait say 10 minutes after pushing to dockerhub, not sure why!
