import click
import numpy as np
import sys

from stempy import io, image

@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-d', '--dark-sample', help='Dark sample file')
@click.option('-r', '--rows', default=160, help='Number of rows')
@click.option('-c', '--columns', default=160, help='Number of columns')
@click.option('-i', '--inner-radius', default=40, help='Mask inner radius')
@click.option('-u', '--outer-radius', default=288, help='Mask outer radius')
@click.option('-v', '--reader-version', default=1, help='Reader version')
@click.option('-x', '--dark-reader-version', default=1,
              help='Dark sample reader version')
@click.option('--save-raw', is_flag=True, default=False,
              help='Save raw data also')
@click.option('--zip-raw', is_flag=True, default=False,
              help='Zip the raw data that is saved')
@click.option('-o', '--output', default='stem_image.h5', help='Output file')
def make_stem_hdf5(files, dark_sample, rows, columns, inner_radius,
                   outer_radius, reader_version, dark_reader_version, save_raw,
                   zip_raw, output):
    """Make an HDF5 file containing a STEM image

    Example: "python create_hdf5.py -d darksample.dat /path/to/data/data*.dat"

    """

    if len(files) == 0:
        sys.exit('No files found')

    if reader_version == 1:
        reader_version = io.FileVersion.VERSION1
    elif reader_version == 2:
        reader_version = io.FileVersion.VERSION2
    else:
        sys.exit('Unknown reader version:', reader_version)

    if dark_reader_version == 1:
        dark_reader_version = io.FileVersion.VERSION1
    elif dark_reader_version == 2:
        dark_reader_version = io.FileVersion.VERSION2
    else:
        sys.exit('Unknown dark reader version:', dark_reader_version)

    reader = io.reader(dark_sample, version=dark_reader_version)
    dark = image.calculate_average(reader)

    reader = io.reader(files, version=reader_version)
    frame_events = image.electron_count(reader, rows, columns, dark)

    # Read one block in to get the detector frames
    reader.reset()
    block = reader.read()
    detector_nx = block.header.columns
    detector_ny = block.header.rows

    reader.reset()
    img = image.create_stem_image(reader, rows, columns, inner_radius,
                                  outer_radius);

    io.save_electron_counts(output, frame_events, rows, columns, detector_nx,
                            detector_ny)
    io.save_stem_image(output, img)

    if save_raw:
        reader.reset()
        blocks = [block for block in reader]

        raw_data = np.concatenate([block.data for block in blocks])
        io.save_raw_data(output, raw_data, zip_raw)

if __name__ == '__main__':
    make_stem_hdf5()
