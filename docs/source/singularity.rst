
Running in singularity
======================

Singularity containers can encapulate the entire stempy environment so that
you don't have to build it. We provide a container that has a Python 3.7 environment
with stempy pre-installed.


Install singularity (Ubuntu)
----------------------------

``sudo apt-get install singularity-container``


Pulling the latest image
------------------------

``singularity pull docker://openchemistry/stempy``


Running a script inside the container
-------------------------------------

By default singularity will mount your home directory into the container. So
for example if you have a script called ``test.py`` in your home directory you
can execute it in the container with the following command:

``singularity exec stempy/stempy.simg python3 ~/test.py``

Mounting other directories into the container
---------------------------------------------

You can also bind mount other directories into the container. For example if
your script reference data at for example ``/mnt/nvmedata1/temp/``, you can mount
with the following command:

``singularity exec -B /mnt/nvmedata1/temp/:/mnt/nvmedata1/temp/ stempy.simg python3 ~/test.py``
