import pytest

import numpy as np

from stempy import io
from stempy.io.sparse_array import FullExpansionDenied
from stempy.io.sparse_array import SparseArray


@pytest.fixture
def sparse_array_small(electron_data_small):
    data = io.load_electron_counts(electron_data_small)
    kwargs = {
        'data': data.data,
        'scan_shape': data.scan_dimensions,
        'frame_shape': data.frame_dimensions,
        'dtype': np.uint64,
    }
    return SparseArray(**kwargs)


def test_full_expansion(sparse_array_small):
    array = sparse_array_small

    # Have to change these so we won't return a SparseArray,
    # and allow it to return a fully expanded array
    array.sparse_slicing = False
    array.allow_full_expand = True
    full = array[:]

    assert isinstance(full, np.ndarray)
    assert full.shape == array.shape


def test_deny_full_expansion(sparse_array_small):
    array = sparse_array_small
    array.sparse_slicing = False
    with pytest.raises(FullExpansionDenied):
        array[:]
