import click
import sys

import numpy as np
from stempy import io, image

@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-r', '--rows', default=160, help='Number of rows')
@click.option('-c', '--columns', default=160, help='Number of columns')
@click.option('-i', '--inner-radius', default=40, help='Mask inner radius')
@click.option('-u', '--outer-radius', default=288, help='Mask outer radius')
@click.option('-v', '--reader-version', default=1, help='Reader version')
@click.option('-o', '--output', default='stem_image.h5', help='Output file')
def make_stem_hdf5(files, rows, columns, inner_radius, outer_radius,
                   reader_version, output):
    """Make an HDF5 file containing raw STEM image data

    Example: "python create_hdf5.py /path/to/data/data*.dat"

    """

    if len(files) == 0:
        sys.exit('No files found')

    if reader_version == 1:
        reader_version = io.FileVersion.VERSION1
    elif reader_version == 2:
        reader_version = io.FileVersion.VERSION2
    else:
        sys.exit('Unknown reader version:', reader_version)

    reader = io.reader(files, version=reader_version)
    blocks = [block for block in reader]

    raw_data = np.concatenate([block.data for block in blocks])

    reader.reset()
    img = image.create_stem_image(reader, rows, columns, inner_radius,
                                  outer_radius);

    io.save_stem_image(output, img)
    io.save_raw_data(output, raw_data)

if __name__ == '__main__':
    make_stem_hdf5()