
Running in singularity
======================

Singularity or Docker containers can encapsulate the entire stempy environment so that
you don't have to build it. We provide a container that has a Python 3.7 environment
with stempy pre-installed. This container will work in Docker, Singularity,
and NERSC's Shifter.


Install Singularity (Ubuntu)
----------------------------

``sudo apt-get install singularity-container``

<<<<<<< HEAD
*Note:* Version 3.61 or higher of Singularity is required to avoid an issue with `squashfs`.
You may need to build the newest version if the apt repositories version does not work.
=======
Note
####

Version 2.4.2-dist on Ubuntu has a bug with squashfs. Version 3.6.1
has been tested to work. Follow installation instructions here:
https://github.com/hpcng/singularity/blob/master/INSTALL.md

Then you will run ``/usr/local/bin/singularity``
>>>>>>> upstream-master

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

Creating and Connecting to a Jupyter Lab Server
-----------------------------------------------

Using Jupyter Lab (or notebook) is very useful for interactively working
with stempy.

On your local machine create a ssh tunnel:

``ssh username@server_name -L 7777:localhost:8888``

On the remote machine start up the Jupyter server:

``singularity exec -B /mnt:/mnt,/tmp:/home/jovyan ~/stempy-scipy-notebook-latest.simg jupyter lab``

The remote machine will print out a message similar to:

.. code-block::

 The Jupyter Notebook is running at:
 http://server_name:8888/?token=
 or http://127.0.0.1:8888/?token=

Copy the entire ``http://127.0.0.1:8888/?token=`` address including the long string of letters and numbers
after ``token=`` which is the secret key for this server.

On your local machine paste the address into your browser of choice.

AND THEN

change the ``:8888`` to ``:7777``

Jupypter Lab will start up in your browser and all calculations will
be run on the remote server.