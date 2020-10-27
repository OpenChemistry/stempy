import sys
from pathlib import Path

import click

import stempy.io as stio
import stempy.image as stim


@click.command()
@click.option('-i', '--input-path', help='HDF5 file containing the electron counts',
              type=click.Path(exists=True, dir_okay=False), default=None, show_default=True)
@click.argument('scan-num', required=False)
def main(input_path, scan_num):
    if input_path is None:
        if scan_num is None:
            raise click.ClickException('Please provide scan number')
        input_path = Path(f'/mnt/hdd1/data_scan{scan_num}_th4_electrons.h5')

    # Load the electron counted data
    ee = stio.load_electron_counts(str(input_path))

    # Create STEM images with inner and outer radii
    ims = stim.create_stem_images(ee, (0, 0, 110, 220),
                                    (110, 220, 240, 288),
                                center=(307, 282))

    # Calculate summed diffraction pattern
    dp = stim.calculate_sum_sparse(ee.data, ee.frame_dimensions)

if __name__ == '__main__':
    main()