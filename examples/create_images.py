import sys
from pathlib import Path

import stempy.io as stio
import stempy.image as stim

scanNum = sys.argv[1]

scanPath = Path('/mnt/hdd1/data_scan{}_th4_electrons.h5'.format(scanNum))

# Load the electron counted data
ee = stio.load_electron_counts(str(scanPath))

# Create STEM images with inner and outer radii
ims = stim.create_stem_images(ee, (0, 0, 110, 220),
                                  (110, 220, 240, 288),
                              center=(307, 282))

# Calculate summed diffraction pattern
dp = stim.calculate_sum_sparse(ee.data, ee.frame_dimensions)
