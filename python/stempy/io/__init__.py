from collections import namedtuple
import numpy as np
import h5py

from stempy._io import _reader

class FileVersion(object):
    VERSION1 = 1
    VERSION2 = 2

class Reader(_reader):
    def __iter__(self):
        return self

    def __next__(self):
        b = self.read()
        if b is None:
            raise StopIteration
        else:
            return b

    def read(self):
        b = super(Reader, self).read()

        # We are at the end of the stream
        if b.header.version == 0:
            return None

        block = namedtuple('Block', ['header', 'data'])
        block._block = b
        block.header = b.header
        block.data = np.array(b, copy = False)

        return block

def reader(path, version=FileVersion.VERSION1):
    return Reader(path, version)

def save_raw_data(path, data, zip_data=False):
    with h5py.File(path, 'a') as f:
        group = f.require_group('stem')

        if zip_data:
            # Make each chunk the size of a frame
            chunk_shape = (1, data.shape[1], data.shape[2])
            group.create_dataset('frames', data=data, compression='gzip',
                                 chunks=chunk_shape)
        else:
            group.create_dataset('frames', data=data)

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

def save_stem_image(outputFile, image):
    with h5py.File(outputFile, 'a') as f:
        stem_group = f.require_group('stem')
        stem_group.create_dataset('bright', data=image.bright)
        stem_group.create_dataset('dark', data=image.dark)
