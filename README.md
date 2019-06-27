Toolkit for processing 4D STEM image data on HPC.

Build instructions may be found [here](BUILDING.md).

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
