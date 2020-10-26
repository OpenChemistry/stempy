
Running in singularity
======================

Singularity or Docker containers can encapsulate the entire stempy environment so that
you don't have to build it. We provide a container that has a Python 3.7 environment
with stempy pre-installed. This container will work in Docker, Singularity,
and NERSC's Shifter.


Install Singularity (Ubuntu)
----------------------------

``sudo apt-get install singularity-container``

*Note:* Version 3.61 or higher of Singularity is required to avoid an issue with `squashfs`.
You may need to build the newest version if the apt repositories version does not work.

Pulling the latest image
------------------------

Use `singularity` or `docker` interchangeably below.

``singularity pull docker://openchemistry/stempy``

``singularity pull docker://openchemistry/stempy-scipy-notebook``


Running a script inside the container
-------------------------------------

By default singularity will mount your home directory into the container. So
for example if you have a script called ``test.py`` in your home directory you
can execute it in the container with the following command:

``singularity exec stempy/stempy.simg python3 ~/test.py``

Running a Jupyter Lab instance inside the container. The -B option mounts directories
on the host to directories inside the notebook.

``singularity exec -B /tmp:/home/jovyan,/mnt:/mnt stempy-scipy-notebook_latest.sif jupyter lab``

Mounting other directories into the container
---------------------------------------------

You can also bind mount other directories into the container. For example if
your script reference data at for example ``/mnt/nvmedata1/temp/``, you can mount
with the following command:

``singularity exec -B /mnt/nvmedata1/temp/:/mnt/nvmedata1/temp/ stempy.simg python3 ~/test.py``
