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
└── images
frames
```

- `electron_events/frames` - Array of arrays, the first index is the scan position
( corresponding to `scan_positions` ) and the second array holds an array of
of indices into the diffractogram where an electron strike was detected.
  - Attributes:
    - `Nx`: the width of the frame
    - `Ny`: the height of the frame

- `electron_events/scan_positions` - Array of shorts holding the scan positions.
  - Attributes:
    - `Nx`: the width of the scan
    - `Ny`: the height of the scan

- `stem/images` - A list of 2D arrays of unsigned integers containing stem images
(possibly bright and dark fields).
  - Attributes:
    - `names`: a list of names assigned to the images

- `frames` - If present, the raw diffractogram data, where the first index is the scan
position index for the 2D diffractogram.

This file can be produced with the script at `examples/create_hdf5.py`
by using a command similar to the following:
```
python create_hdf5.py -h 40 -w 40 -v 2 -x 2 -d /data/4dstem/electronCounting/stem4d_0000000235_0000000001.dat /data/4dstem/electronCounting/stem4d_0000000236_0000000009.dat --save-raw
```

Use `python create_hdf5.py --help` for more information about the
options.
