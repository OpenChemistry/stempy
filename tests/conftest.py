import copy
import io

import numpy as np
import pytest
import requests

from stempy.io.sparse_array import SparseArray


DATA_URLS = {
    'electron_small': 'https://data.kitware.com/api/v1/file/6065f00d2fa25629b93bdabe/download',  # noqa
    'electron_large': 'https://data.kitware.com/api/v1/file/6065f2792fa25629b93c0303/download',  # noqa
    'cropped_multi_frames_v1': 'https://data.kitware.com/api/v1/file/623119814acac99f4261aa59/download',  # noqa
    'cropped_multi_frames_v2': 'https://data.kitware.com/api/v1/file/6244e9944acac99f424743df/download',  # noqa
    'cropped_multi_frames_v3': 'https://data.kitware.com/api/v1/file/624601cd4acac99f425c73f6/download',  # noqa
}

DATA_RESPONSES = {}


def response(key):
    if key not in DATA_RESPONSES:
        r = requests.get(DATA_URLS[key])
        r.raise_for_status()
        DATA_RESPONSES[key] = r

    return DATA_RESPONSES[key]


def io_object(key):
    r = response(key)
    return io.BytesIO(r.content)


@pytest.fixture
def electron_data_small():
    return io_object('electron_small')


@pytest.fixture
def electron_data_large():
    return io_object('electron_large')


@pytest.fixture
def cropped_multi_frames_data_v1():
    return io_object('cropped_multi_frames_v1')


@pytest.fixture
def cropped_multi_frames_data_v2():
    return io_object('cropped_multi_frames_v2')


@pytest.fixture
def cropped_multi_frames_data_v3():
    return io_object('cropped_multi_frames_v3')


# SparseArray fixtures

@pytest.fixture
def sparse_array_10x10():
    aa = np.zeros((4, 3), dtype='object')
    for ii in range(4):
        aa[ii, 0] = np.array((0, 48, 72, 65))
        aa[ii, 1] = np.array((0, 48, 72, 65))
        aa[ii, 2] = np.array((0, 48, 72, 65))

    return SparseArray(aa, (2, 2), (11, 11))

@pytest.fixture
def sparse_array_small(electron_data_small):
    kwargs = {
        'dtype': np.uint64,
    }
    array = SparseArray.from_hdf5(electron_data_small, **kwargs)

    # Perform some slicing so we don't blow up CI memory when we
    # do a full expansion.
    return array[:40:2, :40:2]


cached_full_array_small = None


@pytest.fixture
def full_array_small(sparse_array_small):
    global cached_full_array_small

    if cached_full_array_small is None:
        # Don't allow this fixture to modify the other fixture
        array = copy.deepcopy(sparse_array_small)

        # Have to change these so we won't return a SparseArray,
        # and allow it to return a fully expanded array
        array.sparse_slicing = False
        array.allow_full_expand = True

        cached_full_array_small = array[:]

    return cached_full_array_small


@pytest.fixture
def cropped_multi_frames_v1(cropped_multi_frames_data_v1):
    return SparseArray.from_hdf5(cropped_multi_frames_data_v1, dtype=np.uint16)


@pytest.fixture
def cropped_multi_frames_v2(cropped_multi_frames_data_v2):
    return SparseArray.from_hdf5(cropped_multi_frames_data_v2, dtype=np.uint16)


@pytest.fixture
def cropped_multi_frames_v3(cropped_multi_frames_data_v3):
    return SparseArray.from_hdf5(cropped_multi_frames_data_v3, dtype=np.uint16)

@pytest.fixture
def simulate_sparse_array():
    
    """Make a ndarray with sparse disk.
    scan_size: Real space scan size (pixels)
    frame_size: detector size (pixels)
    center: the center of the disk in diffraction space
    how_sparse: Percent sparseness (0-1). Large number means fewer electrons per frame
    disk_size: The radius in pixels of the diffraction disk
    
    """ 
    
    scan_size = (100, 100)
    frame_size = (100,100)
    center = (30, 70)
    how_sparse = .8 # 0-1; larger is less electrons
    disk_size = 10 # disk radius in pixels
    num_frames = 2
    
    YY, XX = np.mgrid[0:frame_size[0], 0:frame_size[1]]
    RR = np.sqrt((YY-center[1])**2 + (XX-center[0])**2)

    RR[RR <= disk_size] = 1
    RR[RR > disk_size] = 0

    sparse = np.random.rand(scan_size[0],scan_size[1],frame_size[0],frame_size[1], num_frames) * RR[:,:,None]

    sparse[sparse >= how_sparse] = 1
    sparse[sparse < how_sparse] = 0

    sparse = sparse.astype(np.uint8)
    
    new_data = np.empty((sparse.shape[0],sparse.shape[1], num_frames), dtype=object)
    for j in range(sparse.shape[0]):
        for k in range(sparse.shape[1]):
            for i in range(num_frames):
                events = np.ravel_multi_index(np.where(sparse[j,k,:,:,i] == 1),frame_size)
                new_data[j,k,i] = events

    new_data = new_data.reshape(new_data.shape[0]*new_data.shape[1],num_frames)

    kwargs = {
        'data': new_data,
        'scan_shape': sparse.shape[0:2],
        'frame_shape': sparse.shape[2:4],
        'dtype': sparse.dtype,
        'sparse_slicing': True,
        'allow_full_expand': False,
    }
    
    return SparseArray(**kwargs)