import copy

import pytest

import numpy as np

from stempy.io.sparse_array import FullExpansionDenied
from stempy.io.sparse_array import SparseArray


cached_full_array_small = None


@pytest.fixture
def sparse_array_small(electron_data_small):
    kwargs = {
        'dtype': np.uint64,
    }
    array = SparseArray.from_hdf5(electron_data_small, **kwargs)

    # Perform some slicing so we don't blow up CI memory when we
    # do a full expansion.
    return array[:40:2, :40:2]


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

    if array.ndim == 3:
        slices = (0,)
    elif array.ndim == 4:
        slices = (0, 0)

    first_frame = array[slices]

    assert np.array_equal(first_frame, full[slices])

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
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 7), (0, 9), (0, 11),
        (1, 0), (1, 3), (2, 2), (3, 1), (8, 4), (9, 6), (10, 13), (17, 17),
        (14, 16), (18, 15),
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
    shape = (10, 40, 576, 576)
    # Try tuple
    array.reshape(shape)
    assert array.shape == shape

    # Try variable length argument
    array.reshape(*original_shape)
    assert array.shape == original_shape

    # Try a few different reshapings, and test the first frame each time
    shapes = [
        (40, 10, 576, 576),
        (10, 40, 576, 576),
        (1, 400, 576, 576),
        (400, 576, 576),
        (5, 80, 288, 1152),
        (80, 5, 1152, 288),
        (10, 40, 192, 1728),
        (20, 20, 1728, 192),
    ]

    for shape in shapes:
        array = array.reshape(shape)
        full = full.reshape(shape)

        assert array.shape == shape
        assert full.shape == shape
        test_first_frame_expansion(array, full)


def test_index_error(sparse_array_small):
    array = sparse_array_small
    with pytest.raises(IndexError):
        array[array.shape[0]]

    with pytest.raises(IndexError):
        array[:, array.shape[1] + 5]

    with pytest.raises(IndexError):
        array[:, :, :, array.shape[3]]

    with pytest.raises(IndexError):
        array[-array.shape[0] - 1]

    with pytest.raises(IndexError):
        array[:, -array.shape[1] - 5]


def test_slice_shapes(sparse_array_small, full_array_small):
    # Indexing should always remove the axis.
    # Slicing should always retain the axis.
    # To make it simple, just make sure it matches the behavior
    # of numpy...
    array = sparse_array_small
    full = full_array_small

    def run_it(sparse_slicing):
        array.sparse_slicing = sparse_slicing
        assert array[0, :, :, :].shape == full[0, :, :, :].shape
        assert array[0:1, :, :, :].shape == full[0:1, :, :, :].shape
        assert array[0:2, :, :, :].shape == full[0:2, :, :, :].shape

        assert array[:, 0:, :, :].shape == full[:, 0:, :, :].shape
        assert array[:, 0:1, :, :].shape == full[:, 0:1, :, :].shape
        assert array[:, 0:2, :, :].shape == full[:, 0:2, :, :].shape

        assert array[:, :, 0, :].shape == full[:, :, 0, :].shape
        assert array[:, :, 0:1, :].shape == full[:, :, 0:1, :].shape
        assert array[:, :, 0:2, :].shape == full[:, :, 0:2, :].shape

        assert array[:, :, :, 0].shape == full[:, :, :, 0].shape
        assert array[:, :, :, 0:1].shape == full[:, :, :, 0:1].shape
        assert array[:, :, :, 0:2].shape == full[:, :, :, 0:2].shape

    run_it(True)
    run_it(False)


def test_slice_sum(sparse_array_small, full_array_small):
    data = sparse_array_small
    full = full_array_small

    assert np.array_equal(data[0, 0:1, :, :].sum(axis=0),
                          full[0, 0:1, :, :].sum(axis=0))
    assert np.array_equal(data[0:1, 0:1, :, :].sum(axis=(0, 1)),
                          full[0:1, 0:1, :, :].sum(axis=(0, 1)))
    assert np.array_equal(data[0:2, 0:2, :, :].sum(axis=(0, 1)),
                          full[0:2, 0:2, :, :].sum(axis=(0, 1)))
    assert np.array_equal(data[0, 0, :, :].sum(),
                          full[0, 0, :, :].sum())


def test_scan_ravel(sparse_array_small, full_array_small):
    array = sparse_array_small
    full = full_array_small

    array.ravel_scans()
    expected_shape = (400, 576, 576)
    assert array.shape == expected_shape

    # Perform a few simple tests on the raveled SparseArray
    assert np.array_equal(array.sum(axis=0), full.sum(axis=(0, 1)))

    full = full.reshape(expected_shape)
    for i in range(expected_shape[0]):
        assert np.array_equal(array[i], full[i])


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
    (19, 7, -5, 8),
    (3, 4, slice(100, 1, -2)),
    (slice(3, 4, None), slice(70, 10, -2)),
    (slice(None), 5),
    (slice(None), 8, slice(None), slice(1, 300, 3)),
    (slice(4), slice(7), slice(3), slice(5)),
    (slice(2, 8), slice(5, 9), slice(2, 20), slice(50, 200)),
    (slice(2, 18, 2), slice(3, 20, 3), slice(2, 50, 2), slice(20, 100)),
    (slice(None, None, 2), slice(None, None, 3), slice(None, None, 4),
     slice(None, None, 5)),
    (slice(3, None, 2), slice(5, None, 5), slice(4, None, 4),
     slice(20, None, 5)),
    (slice(None, None, -1), slice(20, 4, -2), slice(4, None, -3),
     slice(100, 3, -5)),
]
