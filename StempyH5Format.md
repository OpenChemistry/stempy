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

Where `frames` is an array of arrays,
`scan_positions` is an array of shorts,
and `bright` and `dark` are 2D arrays of
unsigned integers.

This file can be produced with the script at `examples/create_hdf5.py`
by using a command similar to the following:
```
python create_hdf5.py -r 40 -c 40 -v 2 -x 2 -d ~/data/stempy/4dstem/electronCounting/stem4d_0000000235_0000000001.dat ~/data/stempy/4dstem/electronCounting/stem4d_0000000236_0000000009.dat
```

Use `python create_hdf5.py --help` for more information about the
options.
