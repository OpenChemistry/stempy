#!/usr/bin/env python3

from pathlib import Path
import sys
import time

import numpy as np

import stempy.io as stio
import stempy.image as stim

scanNum = sys.argv[1]
scanName = 'data_scan{}_'.format(scanNum)

threshold = int(sys.argv[2])

#  Setup the data drives
drives = []
for ii in range(1,5):
    drives.append( (Path('/mnt/nvmedata{}/'.format(ii))))

print('Looking for files in:')
for d in drives:
    print(d)

dark0 = np.zeros((576,576))

iFiles = []
for drive in drives:
    files = drive.glob(scanName + '*.data')
    for f in files:
        iFiles.append(str(f))

iFiles = sorted(iFiles)

# Electron count the data
sReader = stio.reader(iFiles,stio.FileVersion.VERSION4, backend='thread')

print('start counting')
t0 = time.time()
ee = stim.electron_count(sReader,dark0,number_of_samples=1200,
                                            verbose=False,threshold_num_blocks=20,
                                            xray_threshold_n_sigma=175,
                                            background_threshold_n_sigma=threshold)

frame_events = ee.data

t1 = time.time()
print('total time = {}'.format(t1-t0))

outPath = Path('/mnt/hdd1/data_scan{}_th{}_electrons.h5'.format(scanNum, threshold))
ii = 0
while outPath.exists():
    ii += 1
    outPath = Path(outPath.stem + '_{:03d}'.format(ii))
print('Saving to {}'.format(outPath))
stio.save_electron_counts(str(outPath), ee)


