#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scan_number','-s', type=int)
parser.add_argument('--threshold', '-t', type=float)
parser.add_argument('--num-threads','-r',dest='num_threads', type=int, default=20) # not implemented yet on ms6
parser.add_argument('--location', '-l', type=str, default='/mnt/nvmedata')
parser.add_argument('--no-threads', '-d', dest='threaded',action='store_false') # disable threading
parser.add_argument('--temp-dir', dest='temp_dir', action='store_true')
parser.add_argument('--no-temp-dir', dest='temp_dir', action='store_false')
parser.add_argument('--out-dir', dest='out_dir', type=str)
parser.set_defaults(temp_dir=False, threaded=True)
args = parser.parse_args()

from pathlib import Path
import sys
import time
from datetime import datetime

import numpy as np
import h5py

import stempy.io as stio
import stempy.image as stim

# Inputs
scanNum = args.scan_number
threshold = float(args.threshold)
num_threads = int(args.num_threads)
drive_prefix = args.location
temp_dir = args.temp_dir # Indicates data is in temp directory on NVME drive
out_dir = Path(args.out_dir)

# Check that output directory exists
if not out_dir.is_dir():
    raise FileNotFoundError('Output directory either does not exist or is a file')

if args.threaded:
    print('Using threaded counting code')
    threaded = 'thread'
else:
    print('unthreaded')
    threaded = None

scanName = 'data_scan{:010}_'.format(scanNum)
#scanName = 'data_scan{}_'.format(scanNum)

#  Setup the data drive paths
drives = []
for ii in range(1,5):
    #drives.append( (Path('/mnt/nvmedata{}/temp'.format(ii))))
    drive_name = drive_prefix + '{}'.format(ii)
    if temp_dir:
        # Temporarily staged files
        drive_name += '/temp'
    drives.append(Path(drive_name))

print('Looking for files in:')
for d in drives:
    print(d)

dark0 = np.zeros((576, 576))
gain0 = np.ones((576, 576),dtype=np.float32)
gain = None

iFiles = []
for drive in drives:
    files = drive.glob(scanName + '*.data')
    for f in files:
        iFiles.append(str(f))

# Sort the files
iFiles = sorted(iFiles)

print('Number of files = {}'.format(len(iFiles)))

# Electron count the data
#print('NOTE: Using file version 4')
sReader = stio.reader(iFiles,stio.FileVersion.VERSION5, backend=threaded)

print('start counting')
t0 = time.time()
ee = stim.electron_count(sReader,dark0,gain=gain0,number_of_samples=1200,
                                            verbose=True,threshold_num_blocks=20,
                                            xray_threshold_n_sigma=175,
                                            background_threshold_n_sigma=threshold)

t1 = time.time()
full_time = t1 - t0
print('total time = {}'.format(full_time))

outPath = out_dir / Path('data_scan{}_th{}_electrons.h5'.format(scanNum, threshold))
ii = 0

# Test for existence of the file and change name instead of overwriting
if outPath.exists():
    outPath2 = outPath
    while outPath2.exists():
        ii += 1
        outPath2 = outPath.with_name(outPath.stem + '_{:03d}.h5'.format(ii))
else:
    outPath2 = outPath

# Add meta data
ee.metadata.update({
    'user': {
        'threshold sigma': threshold,
        'scan number': scanNum,
        'threaded': args.threaded,
        'date processed': datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
        'process time (s)': full_time,
    }
})

print('Saving to {}'.format(outPath2))
stio.save_electron_counts(str(outPath2), ee)
print('done')
