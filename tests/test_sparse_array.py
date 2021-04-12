import pytest

import numpy as np

from stempy import io
from stempy.io.sparse_array import FullExpansionDenied
from stempy.io.sparse_array import SparseArray


cached_full_array_small = None


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


@pytest.fixture
def full_array_small(sparse_array_small):
    global cached_full_array_small

    if cached_full_array_small is None:
        array = sparse_array_small

        # Have to change these so we won't return a SparseArray,
        # and allow it to return a fully expanded array
        array.sparse_slicing = False
        array.allow_full_expand = True

        cached_full_array_small = array[:]

    return cached_full_array_small


def test_full_expansion(sparse_array_small, full_array_small):
    array = sparse_array_small
    full = full_array_small

    assert isinstance(full, np.ndarray)
    assert full.shape == array.shape


def test_deny_full_expansion(sparse_array_small):
    array = sparse_array_small
    array.sparse_slicing = False
    with pytest.raises(FullExpansionDenied):
        array[:]


def test_first_frame_expansion(sparse_array_small, full_array_small):
    array = sparse_array_small
    full = full_array_small
    first_frame = array[0, 0]

    assert np.array_equal(first_frame, full[0, 0])

    # Check the data and make sure it truly matches
    unique, counts = np.unique(array.data[0], return_counts=True)
    same = np.zeros(first_frame.shape).flatten()
    same[unique] += counts

    assert np.array_equal(first_frame, same.reshape(first_frame.shape))


def test_dense_slicing(sparse_array_small, full_array_small):
    array = sparse_array_small
    full = full_array_small

    array.sparse_slicing = False

    for slices in TEST_SLICES:
        for i in range(len(slices)):
            # Try using just the first slice, then the first and the second,
            # etc. until using all slices.
            s = slices[:i + 1]

            if len(s) == 1 and s[0] == slice(None):
                # Skip over full expansions
                continue

            assert np.array_equal(array[s], full[s])


def test_sparse_slicing(sparse_array_small, full_array_small):
    array = sparse_array_small
    full = full_array_small

    array.sparse_slicing = True
    array.allow_full_expand = True

    # Just test a few indices
    test_frames = [
        (0, 0), (0, 1), (1, 0), (0, 5), (1, 3), (2, 2), (3, 1), (8, 4), (9, 6),
        (10, 100), (25, 90), (65, 37), (200, 30),
    ]
    for slices in TEST_SLICES:
        sliced = array[slices]
        if sliced.ndim < 2:
            # Skip these ones...
            continue

        was_tested = []
        for x, y in test_frames:
            # Just skip it if it is out of bounds
            if x >= sliced.shape[0] or y >= sliced.shape[1]:
                continue

            assert np.array_equal(sliced[x, y], full[slices][x, y])
            was_tested.append((x, y))

        # Make sure at least 5 frames were tested for each one
        assert len(was_tested) >= 5


def test_arithmetic(sparse_array_small, full_array_small):
    array = sparse_array_small
    full = full_array_small

    ops = [np.sum, np.mean, np.max, np.min]
    axes = [(0, 1)]
    for op in ops:
        for axis in axes:
            sparse_result = op(array, axis=axis)
            full_result = op(full, axis=axis)
            dtype = sparse_result.dtype
            assert np.array_equal(sparse_result, full_result.astype(dtype))


def test_bin_scans(sparse_array_small, full_array_small):
    array = sparse_array_small
    full = full_array_small

    shape = array.shape

    test_until = TEST_BINNING_UNTIL
    binned_with = []
    for factor in range(2, test_until):
        if any(x % factor != 0 for x in shape[:2]):
            # It should fail with a value error
            with pytest.raises(ValueError):
                array.bin_scans(factor)
            continue

        binned = array.bin_scans(factor)
        full_binned = full.reshape(shape[0] // factor, factor,
                                   shape[1] // factor, factor,
                                   shape[2], shape[3]).sum(axis=(1, 3))

        # Expand our binned sparse array
        binned.allow_full_expand = True
        binned.sparse_slicing = False

        assert np.array_equal(binned[:], full_binned)

        binned_with.append(factor)

    # Ensure the test ran some of the time
    assert len(binned_with) >= TEST_BINNING_SUCCESS_MIN

    # Ensure the test failed some of the time
    assert len(binned_with) <= test_until - 2 - TEST_BINNING_FAIL_MIN


def test_bin_frames(sparse_array_small, full_array_small):
    array = sparse_array_small
    full = full_array_small

    shape = array.shape

    test_until = TEST_BINNING_UNTIL
    binned_with = []
    for factor in range(2, test_until):
        if any(x % factor != 0 for x in shape[2:]):
            # It should fail with a value error
            with pytest.raises(ValueError):
                array.bin_frames(factor)
            continue

        binned = array.bin_frames(factor)
        full_binned = full.reshape(shape[0], shape[1], shape[2] // factor,
                                   factor, shape[3] // factor,
                                   factor).sum(axis=(3, 5))

        # Expand our binned sparse array
        binned.allow_full_expand = True
        binned.sparse_slicing = False

        assert np.array_equal(binned[:], full_binned)

        binned_with.append(factor)

    # Ensure the test ran some of the time
    assert len(binned_with) >= TEST_BINNING_SUCCESS_MIN

    # Ensure the test failed some of the time
    assert len(binned_with) <= test_until - 2 - TEST_BINNING_FAIL_MIN


def test_reshape(sparse_array_small, full_array_small):
    array = sparse_array_small
    full = full_array_small

    # Ensure that we can pass a tuple or variable length args
    original_shape = array.shape
    shape = (25, 100, 576, 576)
    # Try tuple
    array.reshape(shape)
    assert array.shape == shape

    # Try variable length argument
    array.reshape(*original_shape)
    assert array.shape == original_shape

    # Try a few different reshapings, and test the first frame each time
    shapes = [
        (25, 100, 576, 576),
        (10, 250, 576, 576),
        (5, 500, 576, 576),
        (10, 250, 288, 1152),
        (250, 10, 1152, 288),
        (5, 500, 192, 1728),
        (500, 5, 1728, 192),
    ]

    for shape in shapes:
        array = array.reshape(shape)
        full = full.reshape(shape)

        assert array.shape == shape
        assert full.shape == shape
        test_first_frame_expansion(array, full)


# Test binning until this number
TEST_BINNING_UNTIL = 33

# Ensure some of the binning succeeded and some failed
TEST_BINNING_SUCCESS_MIN = 4
TEST_BINNING_FAIL_MIN = 4

TEST_SLICES = [
    (0, 4),
    (-1, -1),
    (-5, 10),
    (7, -20, 6),
    (30, 7, -5, 8),
    (3, 4, slice(100, 1, -2)),
    (slice(None), 5),
    (slice(None), 8, slice(None), slice(1, 300, 3)),
    (slice(4), slice(7), slice(3), slice(5)),
    (slice(2, 8), slice(5, 9), slice(2, 20), slice(50, 200)),
    (slice(2, 30, 2), slice(3, 27, 3), slice(2, 50, 2), slice(20, 100)),
    (slice(None, None, 2), slice(None, None, 3), slice(None, None, 4),
     slice(None, None, 5)),
    (slice(3, None, 2), slice(5, None, 5), slice(4, None, 4),
     slice(20, None, 5)),
    (slice(None, None, -1), slice(20, 4, -2), slice(4, None, -3),
     slice(100, 3, -5)),
]
