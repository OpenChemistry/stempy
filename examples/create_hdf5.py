import click
import numpy as np
import sys

from stempy import io, image

@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-d', '--dark-sample', help='Dark sample file')
@click.option('-w', '--width', default=160, help='Width of the stem image')
@click.option('-h', '--height', default=160, help='Height of the stem image')
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
def make_stem_hdf5(files, dark_sample, width, height, inner_radius,
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

    scan_dimensions = (width, height)

    if dark_sample:
        reader = io.reader(dark_sample, version=dark_reader_version)
        dark = image.calculate_average(reader)
    else:
        # Get the frame dimensions from a block header, and use zeros
        # for the dark sample.
        reader = io.reader(files, version=reader_version)
        frame_dimensions = reader.read().header.frame_dimensions
        dark = np.zeros(frame_dimensions)

    reader = io.reader(files, version=reader_version)
    data = image.electron_count(reader, dark, scan_dimensions=scan_dimensions)

    frame_events = data.data
    frame_dimensions = data.frame_dimensions

    inner_radii = [0, inner_radius]
    outer_radii = [outer_radius, outer_radius]
    names = ['bright', 'dark']

    reader.reset()
    imgs = image.create_stem_images(reader, inner_radii, outer_radii,
                                    scan_dimensions=scan_dimensions)

    io.save_electron_counts(output, frame_events, scan_dimensions,
                            frame_dimensions)
    io.save_stem_images(output, imgs, names)

    if save_raw:
        reader.reset()

        # In order to avoid two copies of the data, we must allocate
        # space for the large numpy array first.
        raw_data = np.zeros((np.prod(scan_dimensions), frame_dimensions[1],
                            frame_dimensions[0]), dtype=np.uint16)

        # Make sure the frames are sorted in order
        for block in reader:
            for i in range(len(block.header.image_numbers)):
                num = block.header.image_numbers[i]
                raw_data[num] = block.data[i]

        io.save_raw_data(output, raw_data, zip_data=zip_raw)

if __name__ == '__main__':
    make_stem_hdf5()
