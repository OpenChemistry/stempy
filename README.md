[![Documentation Status](https://readthedocs.org/projects/stempy/badge/?version=latest)](https://stempy.readthedocs.io/en/latest/?badge=latest)

Toolkit for processing 4D STEM image data on HPC.

Singularity instructions may be found [here](https://stempy.readthedocs.io/en/latest/singularity.html).

Build instructions may be found [here](https://stempy.readthedocs.io/en/latest/BUILDING.html).

Example usage
-------------

Build repo and set PYTHONPATH:

```bash
export PYTHONPATH=<build dir>/lib/
```


```python

Python 3.6.7 (default, Oct 22 2018, 11:32:17)
[GCC 8.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from stempy import io
>>> r = io.reader('/data/4dstem/smallScanningDiffraction/data0000.dat')
>>> b = r.read()
>>> b.header.images_in_block
32
>>> b.header.image_numbers
[1, 33, 65, 97, 129, 161, 193, 225, 257, 289, 321, 353, 385, 417, 449, 481, 513, 545, 577, 609, 641, 673, 705, 737, 769, 801, 833, 865, 897, 929, 961, 993]
>>> b.data[0]
array([[   0,    0,    0, ...,    0,    0,    0],
       [   0,    0,    0, ...,    0,    0,    0],
       [   0,    0,    0, ...,    0,    0,    0],
       ...,
       [ 932, 1017,  976, ...,  984,  834, 1031],
       [ 928, 1081, 1100, ..., 1020,  985,  969],
       [ 989,  940, 1045, ..., 1010,  959,  887]], dtype=uint16)
>>>

```

```python
# Example of electron counting raw 4d Camera data
>>> from pathlib import Path
>>> import stempy.io as stio
>>> import stempy.image as stim
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> data_path = Path('/mnt/hdd1/')
>>> files = sorted(data_path.glob('data_scan16*.data'))  # raw data files
>>> sReader = stio.reader(files, stio.FileVersion.VERSION4)
>>> events = stim.electron_count(sReader, np.zeros((576,576)))
>>> stio.save_electron_counts('/mnt/hdd1/data_scan16_electrons.h5',
                          events,
                          events.scan_dimensions,
                          frame_dimensions=(576,576))
# Now create a bright field STEM image from the data
>>> bf = stim.create_stem_images(events, 0, 110)
>>> plt.imshow(bf)

# Create a summed diffraction pattern
>>> dp = np.zeros((576,576), dtype='<u4')
>>> for ev in events:
>>>     try:
>>>         xx, yy = np.unravel_index(ev, (576,576))
>>>     except:  # needed for empty frames
>>>         pass
>>>     dp[xx,yy] += 1
>>> plt.imshow(dp)
```
![Brightfield STEM image](https:/url.to.image/image.jpg)
![Brightfield STEM image](https:/url.to.pattern/pattern.jpg)
