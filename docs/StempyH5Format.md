Stempy H5 Format
================

This file is present to keep a record of the current layout of
the stempy H5 format.

Currently, the structure of the format is as follows:
```
electron_events
├── frames
└── scan_positions
stem
├── bright
└── dark
```

`frames` - Array of arrays, the first index is the scan position
( corresponding to `scan_positions` ) and the second array hold an array of
of indexes into the diffractogram where an electron strike was detected.

`scan_positions` - Array of shorts holding the scan positions.

`bright` - 2D array of unsigned integers containing the bright field image.

`dark` - 2D array of unsigned integers containing the dark field image.

This file can be produced with the script at `examples/create_hdf5.py`
by using a command similar to the following:
```
python create_hdf5.py -r 40 -c 40 -v 2 -x 2 -d ~/data/stempy/4dstem/electronCounting/stem4d_0000000235_0000000001.dat ~/data/stempy/4dstem/electronCounting/stem4d_0000000236_0000000009.dat
```

Use `python create_hdf5.py --help` for more information about the
options.
