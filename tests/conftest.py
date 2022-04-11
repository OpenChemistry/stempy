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
