from pathlib import Path

import sys
import time

import numpy as np
import h5py

import stempy.io as stio
import stempy.image as stim

scanNum = sys.argv[1]
scanName = 'data_scan{}.h5'.format(scanNum)

threshold = int(sys.argv[2])

dPath = '/mnt/nvmedata1/temp'
fPath = Path(scanName)

iPath = dPath / fPath

dark0 = np.zeros((576,576))

print('Opening: {}'.format(iPath))
with h5py.File(iPath, 'r') as f0:

    sReader = stio.reader(f0)
    
    print('start counting')
    t0 = time.time()

    ee = stim.electron_count(sReader, dark0, number_of_samples=1200,
                                             verbose=True,
                                             xray_threshold_n_sigma=175,
                                             background_threshold_n_sigma=threshold)
    
    t1 = time.time()

print('total time = {}'.format(t1-t0))

outPath = Path('/mnt/hdd1/data_scan{}_th{}_electrons.h5'.format(scanNum, threshold))
ii = 0
while outPath.exists():
    ii += 1
    outPath = Path(outPath.stem + '_{:03d}'.format(ii))
print('Saving to {}'.format(outPath))
stio.save_electron_counts(str(outPath), ee)
