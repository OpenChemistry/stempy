[![Documentation Status](https://readthedocs.org/projects/stempy/badge/?version=latest)](https://stempy.readthedocs.io/en/latest/?badge=latest)

Toolkit for processing sparse 4D STEM image data.

Singularity instructions may be found [here](https://stempy.readthedocs.io/en/latest/singularity.html).

Build instructions may be found [here](https://stempy.readthedocs.io/en/latest/BUILDING.html).

Example usage
=============

Electron counting raw 4d Camera data
------------------------------------
```python

from pathlib import Path
import stempy.io as stio
import stempy.image as stim
import matplotlib.pyplot as plt
import numpy as np
data_path = Path('/mnt/hdd1/2020.06.03')
files = sorted(data_path.glob('data_scan0*015_*.data'))  # raw data files
sReader = stio.reader(files, stio.FileVersion.VERSION5)
events = stim.electron_count(sReader, np.zeros((576,576),background_threshold_n_sigma=4.0))
stio.save_electron_counts('/mnt/hdd1/data_scan15_electrons.h5', events)
# Now create a bright field STEM image from the data by summing pixels 
# 0 to 110 radially
bf = stim.create_stem_images(events, 0, 35)
# Create a summed diffraction pattern of the entire dataset
dp = stim.calculate_sum_sparse(events.data,
                                   events.frame_dimensions)
fg, ax = plt.subplots(1,2)
ax[0].imshow(bf[0,:,:])
ax[1].imshow(dp)
```

![Brightfield and diffraction pattern](docs/images/Figure_1.png)

Loading counted data
--------------------

Data output after counting is in a stempy.io.sparse_array.SparseArray format. These can also be saved as an HDF5 
file and loaded from a properly formatted HDF5 file. Here is an example of loading a sparse data set and summing
all diffraction patterns. See the examples for more details.

```python
import stempy.io as stio
import matplotlib.pyplot as plt
sa = stio.SparseArray.from_hdf5('data_scan5_electrons.h5')
dp = sa.sum(axis=(0, 1))
plt.figure()
plt.imshow(dp)
```

Advanced usage for raw data
---------------------------
Build repo and set PYTHONPATH:

```bash
export PYTHONPATH=<build dir>/lib/
```

Interact with raw data

```python
import stempy.io as stio
r = stio.reader('/data/4dstem/smallScanningDiffraction/data0000.dat')
b = r.read()
print(b.header.images_in_block)
# 32
print(b.header.image_numbers)
# [1, 33, 65, 97, 129, 161, 193, 225, 257, 289, 321, 353, 385, 417, 449, 481, 513, 545, 577, 609, 641, 673, 705, 737, 769, 801, 833, 865, 897, 929, 961, 993]
print(b.data)
# array([[   0,    0,    0, ...,    0,    0,    0],
#       [   0,    0,    0, ...,    0,    0,    0],
#       [   0,    0,    0, ...,    0,    0,    0],
#       ...,
#       [ 932, 1017,  976, ...,  984,  834, 1031],
#       [ 928, 1081, 1100, ..., 1020,  985,  969],
#       [ 989,  940, 1045, ..., 1010,  959,  887]], dtype=uint16)
```
