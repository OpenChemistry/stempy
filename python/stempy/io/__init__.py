from collections import namedtuple
import numpy as np
import h5py

from stempy._io import _reader, _sector_reader, _pyreader


class FileVersion(object):
    VERSION1 = 1
    VERSION2 = 2
    VERSION3 = 3
    VERSION4 = 4

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

def get_hdf5_reader(h5file):
    # the initialization is at the io.cpp
    dset_frame=h5file['frames']
    dset_frame_shape=dset_frame.shape
    totalImgNum=dset_frame_shape[0]

    dset_stem_shape=h5file['stem/images'].shape
    scanwidth=dset_stem_shape[2]
    scanheight=dset_stem_shape[1]

    blocksize=32
    # construct the consecutive image_numbers if there is no scan_positions data set in hdf5 file
    if("scan_positions" in h5file):
        image_numbers = h5file['scan_positions']
    else:
        image_numbers = np.arange(totalImgNum)

    h5reader = PyReader(dset_frame, image_numbers, scanwidth, scanheight, blocksize, totalImgNum)
    return h5reader


def reader(path, version=FileVersion.VERSION1):
    # check if the input is the hdf5 dataset
    if(isinstance(path, h5py._hl.files.File)):
        reader = get_hdf5_reader(path)
    elif version == FileVersion.VERSION4:
        reader = SectorReader(path)
    else:
        reader = Reader(path, version)

    return reader

def save_raw_data(path, data, zip_data=False):
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

def save_electron_counts(path, events, scan_nx, scan_ny, detector_nx=None, detector_ny=None):
    with h5py.File(path, 'a') as f:
        group = f.require_group('electron_events')
        scan_positions = group.create_dataset('scan_positions', (events.shape[0],), dtype=np.int32)
        # For now just assume we have all the frames, so the event index can
        # be used to derive the scan_postions.
        # TODO: This should be passed to use
        scan_positions[...] = [i for i in range(0, events.shape[0])]
        scan_positions.attrs['Nx'] = scan_nx
        scan_positions.attrs['Ny'] = scan_ny

        coordinates_type = h5py.special_dtype(vlen=np.uint32)
        frames = group.create_dataset('frames', (events.shape[0],), dtype=coordinates_type)
        # Add the frame dimensions as attributes
        if detector_nx is not None:
            frames.attrs['Nx'] = detector_nx
        if detector_ny is not None:
            frames.attrs['Ny'] = detector_ny

        frames[...] = events

def save_stem_images(outputFile, images, names):
    if len(images) != len(names):
        raise Exception('`images` and `names` must be the same length!')

    with h5py.File(outputFile, 'a') as f:
        stem_group = f.require_group('stem')
        dataset = stem_group.create_dataset('images', data=images)
        dataset.attrs['names'] = names

def write_hdf5(path, reader, format=SectorReader.H5Format.Frame):
    reader.to_hdf5(path, format)