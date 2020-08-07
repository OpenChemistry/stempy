#!/usr/bin/env python3

from pathlib import Path
import sys
import time

import click
import numpy as np

import stempy.io as stio
import stempy.image as stim


@click.command()
@click.option('-o', '--output-path', help='Path to HDF5 file to write electron counts',
              type=click.Path(dir_okay=False), default=None, show_default=True)
@click.argument('scan-num')
@click.argument('threshold')
def main(output_path, scan_num, threshold):
    if output_path is None:
        output_path = Path(f'/mnt/hdd1/data_scan{scan_num}_th{threshold}_electrons.h5')

    scanName = f'data_scan{scan_num}_'

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
    sReader = stio.reader(iFiles,stio.FileVersion.VERSION5, backend='thread')

    print('start counting')
    t0 = time.time()
    ee = stim.electron_count(sReader,dark0,number_of_samples=1200,
                                                verbose=False,threshold_num_blocks=20,
                                                xray_threshold_n_sigma=175,
                                                background_threshold_n_sigma=threshold)

    t1 = time.time()
    print('total time = {}'.format(t1-t0))

    ii = 0
    while output_path.exists():
        ii += 1
        output_path = Path(output_path.stem + '_{:03d}'.format(ii))
    print('Saving to {}'.format(output_path))
    stio.save_electron_counts(str(output_path), ee)

if __name__ == '__main__':
    main()