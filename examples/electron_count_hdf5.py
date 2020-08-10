from pathlib import Path

import sys
import time

import numpy as np
import h5py
import click

import stempy.io as stio
import stempy.image as stim


@click.command()
@click.option('-i', '--input-path', help='Path to HDF5 file to scan data from',
              type=click.Path(exists=True, dir_okay=False), default=None, show_default=True)
@click.option('-o', '--output-path', help='Path to HDF5 file to write electron counts',
              type=click.Path(dir_okay=False), default=None, show_default=True)
@click.argument('scan-num')
@click.argument('threshold')
def main(input_path, output_path, scan_num, threshold):
    if input_path is None:
        input_path = Path(f'/mnt/nvmedata1/temp/data_scan{scan_num}.h5')

    if output_path is None:
        output_path = Path(f'/mnt/hdd1/data_scan{scan_num}_th{threshold}_electrons.h5')

    print(output_path)

    dark0 = np.zeros((576,576))

    print('Opening: {}'.format(input_path))
    with h5py.File(input_path, 'r') as f0:

        sReader = stio.reader(f0)

        print('start counting')
        t0 = time.time()

        ee = stim.electron_count(sReader, dark0, number_of_samples=1200,
                                                verbose=True,
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