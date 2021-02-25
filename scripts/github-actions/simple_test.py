import os
import tempfile

import h5py
import numpy as np

from stempy import image, io


def test_simple():
    # Need at least 32 blocks since the blocksize in
    # io.get_hdf5_reader() is hard-coded to that.
    shape = (32, 576, 576)
    data = np.arange(np.prod(shape)).reshape(shape)
    dataset_name = 'frames'

    # Stempy uses uint16. Let's prevent overflow...
    data %= np.iinfo(np.uint16).max
    mean = data.mean(axis=0)

    # Fake stuff that io.get_hdf5_reader() needs...
    fake_scan_name = 'stem/images'
    fake_scan = np.zeros((4, 8, 1))

    temp = tempfile.NamedTemporaryFile(delete=False)
    file_name = temp.name
    try:
        with h5py.File(file_name, 'w') as wf:
            wf.create_dataset(dataset_name, data=data)
            wf.create_dataset(fake_scan_name, data=fake_scan)

        with h5py.File(file_name, 'r') as rf:
            reader = io.get_hdf5_reader(rf)
            average = image.calculate_average(reader)
    finally:
        del temp
        if os.path.exists(file_name):
            os.remove(file_name)

    # These should be equal...
    assert np.array_equal(average, mean)


if __name__ == '__main__':
    test_simple()
