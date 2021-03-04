from collections import namedtuple
import numpy as np
import h5py

from stempy._io import (
    _reader, _sector_reader, _pyreader, _threaded_reader,
    _threaded_multi_pass_reader
)

COMPILED_WITH_HDF5 = hasattr(_sector_reader, 'H5Format')


class FileVersion(object):
    VERSION1 = 1
    VERSION2 = 2
    VERSION3 = 3
    VERSION4 = 4
    VERSION5 = 5

class ReaderMixin(object):
    def __iter__(self):
        return self

    def __next__(self):
        b = self.read()
        if b is None:
            raise StopIteration
        else:
            return b

    def read(self):
        """Read the next block of data from the file.

        :return: The block of data that was read. Includes the header also.
        :rtype: Block (named tuple with fields 'header' and 'data')
        """
        b = super(ReaderMixin, self).read()

        # We are at the end of the stream
        if b.header.version == 0:
            return None

        block = namedtuple('Block', ['header', 'data'])
        block._block = b
        block.header = b.header
        block.data = np.array(b, copy = False)

        return block


class Reader(ReaderMixin, _reader):
    pass

class SectorReader(ReaderMixin, _sector_reader):
    pass

class PyReader(ReaderMixin, _pyreader):
    pass

class SectorThreadedReader(ReaderMixin, _threaded_reader):
    pass

class SectorThreadedMultiPassReader(ReaderMixin, _threaded_multi_pass_reader):
    pass

def get_hdf5_reader(h5file):
    # the initialization is at the io.cpp
    dset_frame=h5file['frames']
    dset_frame_shape=dset_frame.shape
    totalImgNum=dset_frame_shape[0]

    dset_stem_shape=h5file['stem/images'].shape
    scan_dimensions = (dset_stem_shape[2], dset_stem_shape[1])

    blocksize=32
    # construct the consecutive image_numbers if there is no scan_positions data set in hdf5 file
    if("scan_positions" in h5file):
        image_numbers = h5file['scan_positions']
    else:
        image_numbers = np.arange(totalImgNum)

    h5reader = PyReader(dset_frame, image_numbers, scan_dimensions, blocksize, totalImgNum)
    return h5reader


def reader(path, version=FileVersion.VERSION1, backend=None, **options):
    """reader(path, version=FileVersion.VERSION1)

    Create a file reader to read the data.

    :param path: either the path to the file or an open h5py file.
    :type path: str or h5py file
    :param version: the version of the file reader to use (unused for
                    h5py files).
    :type version: version from stempy.io.FileVersion

    :return: The reader for the data.
    :rtype: stempy.io.Reader, stempy.io.SectorReader, or stempy.io.PyReader
    """
    # check if the input is the hdf5 dataset
    if(isinstance(path, h5py._hl.files.File)):
        reader = get_hdf5_reader(path)
    elif version in [FileVersion.VERSION4, FileVersion.VERSION5]:
        if backend == 'thread':
            reader = SectorThreadedReader(path, version, **options)
        elif backend == 'multi-pass':
            if version != FileVersion.VERSION5:
                raise Exception('The multi pass threaded reader only support file verison 5')

            reader = SectorThreadedMultiPassReader(path, **options)
        elif backend is None:
            reader = SectorReader(path, version)
        # Unrecognized backend
        else:
            raise ValueError(f'Unrecongnized backend: "{backend}"')
    else:
        reader = Reader(path, version)

    return reader

def save_raw_data(path, data, scan_positions=None, zip_data=False):
    """Save the raw data to an HDF5 file.

    :param path: path to the HDF5 file.
    :type path: str
    :param data: the raw data to save.
    :type data: numpy.ndarray
    :param scan_positions: the scan positions of each frame. This is
                           only needed if the frames are not sorted.
    :type scan_positions: list of ints
    :param zip_data: whether or not to compress the data with gzip.
    :type zip_data: bool
    """
    # Chunk cache size. Default is 1 MB
    rdcc_nbytes = 1000000

    if zip_data:
        # Make sure the chunk cache is at least the size of one chunk
        chunk_size = data.shape[1] * data.shape[2] * data.dtype.itemsize
        if rdcc_nbytes < chunk_size:
            rdcc_nbytes = chunk_size

    with h5py.File(path, 'a', rdcc_nbytes=rdcc_nbytes) as f:
        if zip_data:
            # Make each chunk the size of a frame
            chunk_shape = (1, data.shape[1], data.shape[2])
            f.create_dataset('frames', data=data, compression='gzip',
                             chunks=chunk_shape)
        else:
            f.create_dataset('frames', data=data)

        if scan_positions is not None:
            f.create_dataset('scan_positions', data=scan_positions)

def save_electron_counts(path, data):
    """Save the electron counted data to an HDF5 file.

    :param path: path to the HDF5 file.
    :type path: str
    :param data: the electron counted data.
    :type data: ElectronCountedData (named tuple with fields 'data',
                'scan_dimensions', and 'frame_dimensions')
    """
    # Electron counted data attributes
    ecd_attrs = ['data', 'scan_dimensions', 'frame_dimensions']
    if not all([hasattr(data, x) for x in ecd_attrs]):
        raise Exception('`data` must be electron counted data')

    events = data.data
    scan_dimensions = data.scan_dimensions
    frame_dimensions = data.frame_dimensions

    with h5py.File(path, 'a') as f:
        group = f.require_group('electron_events')
        scan_positions = group.create_dataset('scan_positions', (events.shape[0],), dtype=np.int32)
        # For now just assume we have all the frames, so the event index can
        # be used to derive the scan_postions.
        # TODO: This should be passed to use
        scan_positions[...] = [i for i in range(0, events.shape[0])]
        scan_positions.attrs['Nx'] = scan_dimensions[0]
        scan_positions.attrs['Ny'] = scan_dimensions[1]

        coordinates_type = h5py.special_dtype(vlen=np.uint32)
        frames = group.create_dataset('frames', (events.shape[0],), dtype=coordinates_type)
        # Add the frame dimensions as attributes
        frames.attrs['Nx'] = frame_dimensions[0]
        frames.attrs['Ny'] = frame_dimensions[1]

        frames[...] = events

def load_electron_counts(path):
    """Load electron counted data from an HDF5 file.

    :param path: path to the HDF5 file.
    :type path: str

    :return: the coordinates of the electron hits for each frame.
    :rtype: ElectronCountedData (named tuple with fields 'data',
            'scan_dimensions', and 'frame_dimensions')
    """

    ret = namedtuple('ElectronCountedData',
                     ['data', 'scan_dimensions', 'frame_dimensions'])

    with h5py.File(path, 'r') as f:
        frames = f['electron_events/frames']
        scan_positions = f['electron_events/scan_positions']

        ret.data = frames[()]
        ret.scan_dimensions = [scan_positions.attrs[x] for x in ['Nx', 'Ny']]
        ret.frame_dimensions = [frames.attrs[x] for x in ['Nx', 'Ny']]

    return ret

def save_stem_images(outputFile, images, names):
    """Save STEM images to an HDF5 file.

    :param outputFile: path to the HDF5 file.
    :type outputFile: str
    :param images: an array of STEM images.
    :type images: numpy.ndarray
    :param names: a list of names for the STEM images, to be saved as
                  attributes. Must be the same length as `images`.
    :type names: a list of strings
    """
    if len(images) != len(names):
        raise Exception('`images` and `names` must be the same length!')

    with h5py.File(outputFile, 'a') as f:
        stem_group = f.require_group('stem')
        dataset = stem_group.create_dataset('images', data=images)
        dataset.attrs['names'] = names


if COMPILED_WITH_HDF5:
    def write_hdf5(path, reader, format=SectorReader.H5Format.Frame):
        """write_hdf5(path, reader, format=SectorReader.H5Format.Frame)

        Write the data from a SectorReader to an HDF5 file.

        :param path: path to the output HDF5 file.
        :type path: str
        :param reader: a SectorReader that has opened the data.
        :type reader: stempy.io.SectorReader
        :param format: whether to write in frame format or data cube format.
        :type format: SectorReader.H5Format
        """
        reader.to_hdf5(path, format)
