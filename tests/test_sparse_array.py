import os
import tempfile

import pytest

import numpy as np

from stempy.io.sparse_array import FullExpansionDenied
from stempy.io.sparse_array import SparseArray


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
    concatenated = np.concatenate(array.data[0])
    unique, counts = np.unique(concatenated, return_counts=True)
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


def test_frame_slicing(sparse_array_small, full_array_small):
    array = sparse_array_small
    full = full_array_small

    # These are common use cases we want to make sure are supported
    array.allow_full_expand = True
    array.sparse_slicing = True

    # First test the sparse slicing version
    sliced = array[:, :, 100:200, 200:300]
    sliced.sparse_slicing = False

    assert np.array_equal(sliced[:], full[:, :, 100:200, 200:300])

    # Now test the dense slicing version
    array.sparse_slicing = False
    assert np.array_equal(array[:, :, 100:200, 200:300],
                          full[:, :, 100:200, 200:300])


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

        assert np.array_equal(binned.to_dense(), full_binned)

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

        assert np.array_equal(binned.to_dense(), full_binned)

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

        assert array[:, 0:1, :, :].shape == full[:, 0:1, :, :].shape
        assert array[:, 0:2, :, :].shape == full[:, 0:2, :, :].shape
        assert array[:, 0:3, :, :].shape == full[:, 0:3, :, :].shape

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


def test_round_trip_hdf5(sparse_array_small):
    # Test writing to HDF5 then reading back in
    original = sparse_array_small

    # Store some metadata that we will test for
    original.metadata.update({
        'nested': {
            'string': 's',
        },
        'array1d': np.arange(3, dtype=int),
        'array2d': np.arange(9, dtype=np.float32).reshape((3, 3)),
        'float': np.float64(7.21),
    })

    temp = tempfile.NamedTemporaryFile(delete=False)
    try:
        temp.close()
        original.write_to_hdf5(temp.name)
        array = SparseArray.from_hdf5(temp.name, dtype=np.uint64)
    finally:
        os.remove(temp.name)

    # These should be equal in every respect
    assert original.scan_shape == array.scan_shape
    assert original.frame_shape == array.frame_shape
    assert dicts_equal(original.metadata, array.metadata)

    for sfa, sfb in zip(original.data, array.data):
        for a, b in zip(sfa, sfb):
            assert np.array_equal(a, b)

    original.metadata['float'] = np.float64(6.5)
    # Now, the metadata shouldn't be equal
    assert not dicts_equal(original.metadata, array.metadata)


def dicts_equal(d1, d2):
    if d1.keys() != d2.keys():
        return False

    for k, v1 in d1.items():
        v2 = d2[k]

        if type(v1) != type(v2):
            return False

        if isinstance(v1, np.ndarray):
            if v1.shape != v2.shape:
                return False
            if not np.allclose(v1, v2):
                return False
        elif isinstance(v1, dict):
            if not dicts_equal(v1, v2):
                return False
        else:
            if v1 != v2:
                return False

    return True


def test_multiple_frames_per_scan_position():
    # Some simple tests where we know the answer for multiple frames
    # per scan position.
    data = np.empty((2, 2), dtype=object)
    data[0][0] = np.array([0])
    data[0][1] = np.array([0, 2])
    data[1][0] = np.array([0])
    data[1][1] = np.array([0, 1])
    kwargs = {
        'data': data,
        'scan_shape': (2, 1),
        'frame_shape': (2, 2),
        'sparse_slicing': False,
        'allow_full_expand': True,
    }
    array = SparseArray(**kwargs)

    # These are our expected expansions of the positions
    position_zero = np.array([[2, 0], [1, 0]])
    position_one = np.array([[2, 1], [0, 0]])

    # Test full array
    full = array[:]
    assert full.shape == (2, 1, 2, 2)
    assert np.array_equal(full[0, 0], position_zero)
    assert np.array_equal(full[1, 0], position_one)

    # Test dense slicing
    assert array[0].shape == (1, 2, 2)
    assert np.array_equal(array[0, 0], position_zero)
    assert np.array_equal(array[1, 0], position_one)

    # Test sparse slicing
    array.sparse_slicing = True

    # Scan slicing
    assert isinstance(array[0], SparseArray)
    assert isinstance(array[:, 0], SparseArray)
    assert array[0].shape == full[0].shape
    assert array[:, 0].shape == full[:, 0].shape
    assert np.array_equal(array[0][0], array[0, 0])
    assert np.array_equal(array[1][0], array[1, 0])
    assert np.array_equal(array[0:][0][0], array[0, 0])
    assert np.array_equal(array[1:][0][0], array[1, 0])
    assert np.array_equal(array[:, 0][0], array[0, 0])
    assert np.array_equal(array[:, 0][1], array[1, 0])
    assert np.array_equal(array[::2, 0][0], full[::2, 0][0])

    # Frame slicing
    assert isinstance(array[:, :, 0], SparseArray)
    assert array[:, :, 0].shape == full[:, :, 0].shape
    assert array[:, :, 0, 1].shape == full[:, :, 0, 1].shape
    assert np.array_equal(array[:, :, 0][0, 0], full[:, :, 0][0, 0])
    assert np.array_equal(array[:, :, 0][1, 0], full[:, :, 0][1, 0])
    assert np.array_equal(array[:, :, 0, 1][0, 0], full[:, :, 0, 1][0, 0])
    assert np.array_equal(array[:, :, 0, 1:][0, 0], full[:, :, 0, 1:][0, 0])
    assert np.array_equal(array[:, :, 0:2, 0:2][0, 0],
                          full[:, :, 0:2, 0:2][0, 0])
    assert np.array_equal(array[:, :, 0:2, 0:2][1, 0],
                          full[:, :, 0:2, 0:2][1, 0])
    assert np.array_equal(array[0, :, :, 0][0], full[0, :, :, 0][0])
    assert np.array_equal(array[0, :, :, 1][0], full[0, :, :, 1][0])
    assert np.array_equal(array[0, :, 0:2, 0:2][0], full[0, :, 0:2, 0:2][0])
    assert np.array_equal(array[1, :, 0:2, 0:2][0], full[1, :, 0:2, 0:2][0])
    assert np.array_equal(array[0, 0, ::2, ::2], full[0, 0, ::2, ::2])
    assert np.array_equal(array[1, 0, ::2, ::2], full[1, 0, ::2, ::2])

    # Test arithmetic
    axis = (0, 1)
    assert np.array_equal(array.sum(axis=axis), [[4, 1], [1, 0]])
    assert np.array_equal(array.max(axis=axis), [[2, 1], [1, 0]])
    assert np.array_equal(array.min(axis=axis), [[2, 0], [0, 0]])
    assert np.array_equal(array.mean(axis=axis), [[2, 0.5], [0.5, 0]])

    # Test frame binning
    data = np.empty((2, 2), dtype=object)
    data[0][0] = np.array([0, 2, 3, 5, 9])
    data[0][1] = np.array([15])
    data[1][0] = np.array([7])
    data[1][1] = np.array([], dtype=int)

    kwargs = {
        'data': data,
        'scan_shape': (2, 1),
        'frame_shape': (4, 4),
        'sparse_slicing': False,
    }
    test_bin_frames_array = SparseArray(**kwargs)
    binned = test_bin_frames_array.bin_frames(2)

    assert binned.shape == (2, 1, 2, 2)
    assert np.array_equal(binned[0, 0], [[2, 2], [1, 1]])
    assert np.array_equal(binned[1, 0], [[0, 1], [0, 0]])

    # Test scan binning
    data = np.empty(16, dtype=object)
    data[0] = np.array([0, 1])
    data[1] = np.array([0, 2])
    # Automate the rest of them...
    for i in range(2, 16):
        data[i] = np.array([i % 4])

    kwargs = {
        'data': data,
        'scan_shape': (4, 4),
        'frame_shape': (2, 2),
        'sparse_slicing': False,
        'allow_full_expand': True,
    }
    test_bin_scans_array = SparseArray(**kwargs)
    binned = test_bin_scans_array.bin_scans(2)

    assert binned.shape == (2, 2, 2, 2)
    assert binned.num_frames_per_scan == 4
    assert np.array_equal(binned[0, 0], [[3, 2], [1, 0]])
    assert np.array_equal(binned[0, 1], [[0, 0], [2, 2]])
    assert np.array_equal(binned[1, 0], [[2, 2], [0, 0]])
    assert np.array_equal(binned[1, 1], [[0, 0], [2, 2]])

    # Ensure we can bin again
    binned_twice = binned.bin_scans(2)
    assert binned_twice.shape == (1, 1, 2, 2)
    assert binned_twice.num_frames_per_scan == 16

    # Ensure binning 4x is equal to binning 2x twice
    binned_4x = test_bin_scans_array.bin_scans(4)
    assert np.array_equal(binned_4x[:], binned_twice[:])


def test_data_conversion(cropped_multi_frames_v1, cropped_multi_frames_v2,
                         cropped_multi_frames_v3):
    # These datasets were in older formats, but when they were converted
    # into a SparseArray via the fixture, they should have automatically
    # been converted to the latest version. Confirm this is the case.
    latest_version = 3

    to_test_list = [
        cropped_multi_frames_v1,
        cropped_multi_frames_v2,
        cropped_multi_frames_v3,
    ]
    for array in to_test_list:
        array.allow_full_expand = True

        assert array.VERSION == latest_version

        # Should be no duplicates
        for scan_frames in array.data:
            for sparse_frame in scan_frames:
                assert len(np.unique(sparse_frame)) == len(sparse_frame)

        # From local testing, these values should match
        assert array.min() == 0
        assert array.max() == 2
        assert array.sum() == 949625
        assert np.isclose(array.mean(), 0.0071556185)

        assert array.shape == (20, 20, 576, 576)

        # There should be two frames per scan
        assert array.data.shape == (400, 2)

        assert array.num_scans == 400
        assert array.num_frames_per_scan == 2


def test_advanced_indexing(sparse_array_small, full_array_small):
    array = sparse_array_small
    full = full_array_small

    # First test advanced integer indexing
    to_test = [
        [[0]],
        [[1], [14]],
        [[10], [-1]],
        [[10, 7, 3], [7, 0, 9]],
        [[3, 2, 1], [0, 1, 2]],
        [[-1, 4, -3], [-3, 8, 7]],
        [[7], 4],
        [3, [8]],
        [[-1, 4, -3], slice(2, 18, 2)],
        [3, 4, [3, 2, 1], [9, 8, 7]],
        [slice(None), slice(None), [5, 45, 32], [-3, -7, 23]],
        [slice(None), slice(2, 18, 2), slice(0, 100, 3), [-5, 1, 2]],
    ]

    def compare_with_sparse(full, sparse):
        # This will fully expand the sparse array and perform a comparison
        try:
            if isinstance(sparse, SparseArray):
                # This will be a SparseArray except for single frame cases
                prev_allow_full_expand = sparse.allow_full_expand
                prev_sparse_slicing = sparse.sparse_slicing

                sparse.allow_full_expand = True
                sparse.sparse_slicing = False

            return np.array_equal(full, sparse[:])
        finally:
            if isinstance(sparse, SparseArray):
                sparse.allow_full_expand = prev_allow_full_expand
                sparse.sparse_slicing = prev_sparse_slicing

    # First test sparse slicing
    array.sparse_slicing = True
    for index in to_test:
        index = tuple(index)
        assert compare_with_sparse(full[index], array[index])

    # Now test dense slicing
    array.sparse_slicing = False
    for index in to_test:
        index = tuple(index)
        assert np.array_equal(full[index], array[index])

    # Now test boolean indexing.
    # We'll make some random masks for this.
    rng = np.random.default_rng(0)

    num_tries = 3
    for i in range(num_tries):
        scan_mask = rng.choice([True, False], array.scan_shape)
        frame_mask = rng.choice([True, False], array.frame_shape)

        array.sparse_slicing = True
        assert compare_with_sparse(full[scan_mask], array[scan_mask])
        assert compare_with_sparse(full[scan_mask[0]], array[scan_mask[0]])

        assert compare_with_sparse(full[:, :, frame_mask], array[:, :, frame_mask])
        assert compare_with_sparse(full[:, :, frame_mask[0]], array[:, :, frame_mask[0]])

        array.sparse_slicing = False
        assert np.array_equal(full[scan_mask], array[scan_mask])
        assert np.array_equal(full[scan_mask[0]], array[scan_mask[0]])

        assert np.array_equal(full[:, :, frame_mask], array[:, :, frame_mask])
        assert np.array_equal(full[:, :, frame_mask[0]], array[:, :, frame_mask[0]])

    # These should raise index errors
    with pytest.raises(IndexError):
        # Can't do advanced indexing for both the scan and the frame
        # simultaneously.
        array[scan_mask, frame_mask]

    with pytest.raises(IndexError):
        # Too many dimensions for the scan mask.
        array[scan_mask[:, :, np.newaxis]]

    with pytest.raises(IndexError):
        # We don't allow 2D arrays that aren't boolean masks, because
        # an extra dimension gets added, which doesn't make sense for us.
        array[[[1, 2], [5, 4]]]

    # Now test a few arithmetic operations combined with the indexing
    array.sparse_slicing = True
    assert np.array_equal(array[scan_mask].sum(axis='scan'),
                          full[scan_mask].sum(axis=(0,)))
    assert np.array_equal(array[scan_mask].min(axis='scan'),
                          full[scan_mask].min(axis=(0,)))
    assert np.array_equal(array[scan_mask].max(axis='scan'),
                          full[scan_mask].max(axis=(0,)))
    assert np.allclose(array[scan_mask].mean(axis='scan'),
                       full[scan_mask].mean(axis=(0,)))

    # Now do some basic tests with multiple frames per scan
    data = np.empty((2, 2), dtype=object)
    data[0][0] = np.array([0])
    data[0][1] = np.array([0, 2])
    data[1][0] = np.array([0])
    data[1][1] = np.array([0, 1])
    kwargs = {
        'data': data,
        'scan_shape': (2, 1),
        'frame_shape': (2, 2),
        'sparse_slicing': False,
        'allow_full_expand': True,
    }
    # The multiple frames per scan array
    m_array = SparseArray(**kwargs)

    # These are our expected expansions of the positions
    position_zero = np.array([[2, 0], [1, 0]])
    position_one = np.array([[2, 1], [0, 0]])

    # Verify our assumption is true
    assert np.array_equal(m_array[0, 0], position_zero)
    assert np.array_equal(m_array[1, 0], position_one)

    # Now test some simple fancy indexing
    assert np.array_equal(m_array[[0], [0]][0], position_zero)
    assert np.array_equal(m_array[[0, 1], [0]][0], position_zero)
    assert np.array_equal(m_array[[0, 1], [0]][1], position_one)
    assert np.array_equal(m_array[[1, 0], [0]][0], position_one)
    assert np.array_equal(m_array[[1, 0], [0]][1], position_zero)
    assert np.array_equal(m_array[0, 0, [0, 1], [0, 0]], [2, 1])
    assert np.array_equal(m_array[0, 0, [[True, False], [True, False]]],
                          [2, 1])
    assert np.array_equal(m_array[0, 0, [[False, True], [False, True]]],
                          [0, 0])
    assert np.array_equal(m_array[[True, False], 0][0], position_zero)
    assert np.array_equal(m_array[[False, True], 0][0], position_one)


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
    (slice(20, 4, -2), slice(None, None, -1), slice(4, None, -3),
     slice(100, 3, -5)),
]
