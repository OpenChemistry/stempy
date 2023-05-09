from collections import namedtuple
import numpy as np
import h5py

from stempy._io import (
    _reader, _sector_reader, _pyreader, _threaded_reader,
    _threaded_multi_pass_reader
)

# For exporting SparseArray
from .sparse_array import SparseArray

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
    @property
    def num_frames_per_scan(self):
        """Get the number of frames per scan

        It will be cached if it has already been computed. If not, all of
        the header files must be read (which can take some time), and then
        it will check how many frames each position has.
        """
        return self._num_frames_per_scan()

    @property
    def scan_shape(self):
        """Get the scan shape

        If it hasn't been done already, one header file will be read to
        obtain this info.
        """
        scan_dimensions = self._scan_dimensions()
        # We treat the "shape" as having reversed axes than the "dimensions"
        return scan_dimensions[::-1]

    def read_frames(self, scan_position, frames_slice=None):
        """Read frames from the specified scan position and return them

        The scan_position is either a tuple of a valid position in the
        scan_shape, or an integer that is a flattened index of the position.

        The frames_slice object will be used as an index in numpy to
        determine which frames need to be read. If None, all frames
        will be returned.

        Returns a list of blocks for the associated frames.
        """
        if isinstance(scan_position, (list, tuple)):
            # Unravel the scan position
            scan_shape = self.scan_shape
            if (any(not 0 <= scan_position[i] < scan_shape[i]
                    for i in range(len(scan_position)))):
                raise IndexError(
                    f'Invalid position {scan_position} '
                    f'for scan_shape {scan_shape}'
                )

            image_number = scan_position[0] * scan_shape[1] + scan_position[1]
        else:
            # Should just be an integer representing the image number
            image_number = scan_position

        single_index_frame = False

        # First, get the number of frames per scan
        num_frames = self.num_frames_per_scan

        # Create an arange containing all frame positions
        frames = np.arange(num_frames)

        if frames_slice is not None:
            if isinstance(frames_slice, (int, np.integer)):
                frames_slice = [frames_slice]
                single_index_frame = True

            # Slice into the frames object
            try:
                frames = frames[frames_slice]
            except IndexError:
                msg = (
                    f'frames_slice "{frames_slice}" is invalid for '
                    f'num_frames "{num_frames}"'
                )
                raise IndexError(msg)

        blocks = []

        raw_blocks = self._load_frames(image_number, frames)
        for b in raw_blocks:
            block = namedtuple('Block', ['header', 'data'])
            block._block = b
            block.header = b.header
            block.data = np.array(b, copy=False)[0]
            blocks.append(block)

        return blocks[0] if single_index_frame else blocks


def get_hdf5_reader(h5file):
    # the initialization is at the io.cpp
    dset_frame=h5file['frames']
    dset_frame_shape=dset_frame.shape
    totalImgNum=dset_frame_shape[0]
    scan_dimensions = dset_frame.attrs.get('scan_dimensions')
    if scan_dimensions is None:
        # Must be an older file. Give a warning and fall back to the shape.
        print('WARNING: "scan_dimensions" not found on "/frames"',
              '(which may imply an older file is being loaded).',
              'Falling back to the shape of "stem/images"')
        dset_stem_shape = h5file['stem/images'].shape
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

def save_raw_data(path, data, scan_dimensions=None, scan_positions=None,
                  zip_data=False):
    """Save the raw data to an HDF5 file.

    :param path: path to the HDF5 file.
    :type path: str
    :param data: the raw data to save.
    :type data: numpy.ndarray
    :param scan_dimensions: the dimensions of the scan, where the order is
                            (width, height).
    :type scan_dimensions: tuple of ints of length 2
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
            frames = f.create_dataset('frames', data=data, compression='gzip',
                                      chunks=chunk_shape)
        else:
            frames = f.create_dataset('frames', data=data)

        if scan_dimensions is not None:
            frames.attrs['scan_dimensions'] = scan_dimensions

        if scan_positions is not None:
            f.create_dataset('scan_positions', data=scan_positions)

def save_electron_counts(path, array):
    """Save the electron counted data to an HDF5 file.

    :param path: path to the HDF5 file.
    :type path: str
    :param array: the electron counted data.
    :type array: SparseArray
    """
    array.write_to_hdf5(path)

def load_electron_counts(path):
    """Load electron counted data from an HDF5 file.

    :param path: path to the HDF5 file.
    :type path: str

    :return: a SparseArray containing the electron counted data
    :rtype: SparseArray
    """
    return SparseArray.from_hdf5(path)

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

def get_scan_path(directory, scan_num=None, scan_id=None, th=None):
    """ Get the file path for a 4D Camera scan on NERSC using the scan number, 
    the Distiller scan id, and/or threshold. scan_id should always
    be unique and is the best option to load a dataset.
    
    A ValueError is raised if more than one file matches the input. Then the user
    needs to input more information to narrow down the choices.
    
    Parameters
    ----------
    directory : pathlib.Path or str
        The path to the directory containing the file.
    scan_id : int, optional
        The Distiller scan id.
    scan_num : int, optional
        The 4D Camera scan number. Optional
    th : float, optional
        The threshold for counting. This was added to the filename in older files.
        
    Returns
    -------
    : tuple
        The tuple contains the file that matches the input information and the 
        scan_num and scan_id as a tuple.
    """
    if scan_id is not None:
        # This should be unique
        file_path = list(directory.glob(f'*_id{scan_id}*.h5'))
    elif scan_num is not None:
        # older files might include the threshold (th)
        if th is not None:
            file_path = list(directory.glob(f'data_scan{scan_num}_th{th}_electrons.h5'))
        else:
            file_path = list(directory.glob(f'data_scan{scan_num}*electrons.h5'))
    else:
        raise TypeError('Missing scan_num or scan_id input.')

    if len(file_path) > 1:
        raise ValueError('Multiple files match that input. Add scan_id to be more specific.')
    elif len(file_path) == 1:
        file_path = file_path[0]
        # Determine the scan_id and scan_num for use later (i.e. getting DM4 file)
        spl = file_path.name.split('_')
        for ii in spl:
            if 'id' in ii:
                scan_id = int(ii[len('id'):])
            elif 'scan' in ii:
                scan_num = int(ii[len('scan'):])
    else:
        raise FileNotFoundError('No file with those parameters can be found.')
    return file_path, scan_num, scan_id
