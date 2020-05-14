import sys
from pathlib import Path

import stempy.io as stio
import stempy.image as stim
import numpy as np
import h5py

import imageio

scanNum = sys.argv[1]

scanName = 'data_scan{}'.format(scanNum)

scanPath = Path('/mnt/hdd1/data_scan{}_th4_electrons.h5'.format(scanNum))

# Load from numpy
#ee = np.load(scanPath,allow_pickle = True)
#scanSize = (512, 513) # (slow scan, fast scan)
#print('Loading numpy. Update scan size if necessary')

# Load from HDF5
with h5py.File(scanPath,'r') as f0:
    ee = f0['electron_events/frames'][:]
    scanSize = (f0['electron_events/scan_positions'].attrs['Nx'], f0['electron_events/scan_positions'].attrs['Ny'])

# Create STEM images with inner and outer radii
ims = stim.create_stem_images(ee,(0,0,110,220),(110,220,240,288),scan_dimensions = scanSize,frame_dimensions=(576,576),center=(307,282))

# Transpose to match scanning directions on microsocpe
ims.transpose((0,2,1))

# Save as 3D tif.
imageio.mimsave('/mnt/hdd1/data_scan{}_stemImages.tif'.format(scanNum),ims.transpose((0,2,1)).astype('f'))

dp = np.zeros((576,576),dtype='<u4')
for ev in ee:
    try:
        xx,yy = np.unravel_index(ev,(576,576))
    except:
        pass
    dp[xx,yy] += 1

imageio.imsave('/mnt/hdd1/data_scan{}_dp.tif'.format(scanNum),dp.astype('f'))
