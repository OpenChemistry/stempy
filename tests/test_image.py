import pytest
import time

import numpy as np

from stempy.image import com_dense, com_sparse, radial_sum_sparse, create_stem_images
from stempy.io.sparse_array import SparseArray


@pytest.mark.parametrize("center", [
    (2, 1), (2.0, 1.0), np.array([2, 1]),
    (2.75, 1.25), np.array([[2.75], [1.25]]), (-1, -1),
])
@pytest.mark.parametrize("sparse", [False, True])
def test_create_stem_images_center(center, sparse):
    # Unequal coordinates detect x/y swaps, fractional centers must not truncate.
    frames = np.ones((32, 5, 5), dtype=np.uint16)
    if sparse:
        data = np.empty((32, 1), dtype=object)
        for i in range(32):
            data[i, 0] = np.arange(25, dtype=np.uint32)
        source = SparseArray(data=data, scan_shape=(4, 8), frame_shape=(5, 5))
    else:
        source = frames

    images = create_stem_images(source, [0, 1], [1, 2],
                                scan_dimensions=(8, 4), center=center)
    x, y = np.asarray(center).reshape(2)
    if x < 0:
        x = 3  # Preserve the existing rounded default on odd-sized frames.
    if y < 0:
        y = 3
    yy, xx = np.indices((5, 5))
    distances = (xx - x) ** 2 + (yy - y) ** 2
    expected = [np.count_nonzero((distances >= lo ** 2) & (distances < hi ** 2))
                for lo, hi in [(0, 1), (1, 2)]]

    # Sparse integration sums the existing 0xFFFF mask values per electron.
    if sparse:
        expected = np.array(expected) * 0xFFFF

    assert images.shape == (2, 4, 8)
    np.testing.assert_array_equal(images, np.broadcast_to(
        np.array(expected)[:, None, None], images.shape))


def test_create_stem_images_com_dense_center():
    frame = np.zeros((5, 5), dtype=np.uint16)
    frame[1, 2] = 1
    frame[2, 3] = 3
    center = com_dense(frame)
    np.testing.assert_array_equal(center, [[2.75], [1.75]])
    images = create_stem_images(np.repeat(frame[None], 32, axis=0), 0, 1,
                                scan_dimensions=(8, 4), center=center)
    np.testing.assert_array_equal(images, np.full((1, 4, 8), 3))
    np.testing.assert_array_equal(center, [[2.75], [1.75]])


@pytest.mark.parametrize("center", [(1,), (1, 2, 3), [[1, 2]],
                                     (np.nan, 1), (1, np.inf)])
def test_create_stem_images_invalid_center(center):
    with pytest.raises(ValueError, match='center'):
        create_stem_images(np.ones((32, 5, 5), dtype=np.uint16), 0, 1,
                           scan_dimensions=(8, 4), center=center)


@pytest.mark.parametrize("version", [0, 1])
def test_com_sparse(sparse_array_small, full_array_small, version):
    # Do a basic test of com_sparse() with multiple frames per scan
    # position.
    data = np.empty((2, 2), dtype=object)
    data[0][0] = np.array([0])
    data[0][1] = np.array([3])
    data[1][0] = np.array([0])
    data[1][1] = np.array([0, 1, 2])
    kwargs = {
        'data': data,
        'scan_shape': (2, 1),
        'frame_shape': (2, 2),
    }
    array = SparseArray(**kwargs)

    com = com_sparse(array, version=version)

    assert np.array_equal(com[:, 0], [[0.5], [0.5]])
    assert np.array_equal(com[:, 1], [[0.25], [0.25]])

    # Now, test and make sure the center of mass computed for some real
    # data matches the dense array version.

    com = com_sparse(sparse_array_small, version=version)

    full_com = np.empty((2, *full_array_small.shape[:2]), dtype=np.float32)
    for i in range(full_array_small.shape[0]):
        for j in range(full_array_small.shape[1]):
            full_com[:, i, j] = com_dense(full_array_small[i, j])[:, 0]

    assert np.array_equal(com, full_com)


def test_radial_sum_sparse(sparse_array_10x10):

    rr = radial_sum_sparse(sparse_array_10x10, center=(5, 5))

    assert np.array_equal(rr[0, 0, :], [0, 6, 0, 0, 3])


@pytest.mark.parametrize("version", [0, 1])
def test_com_sparse_parameters(simulate_sparse_array, version):

    sp = simulate_sparse_array #((100,100), (100,100), (30,70), (0.8), (10))

    # Test no inputs. This should be the full frame COM
    com0 = com_sparse(sp, version=version)
    assert round(com0[0,].mean()) == 30

    # Test crop_to input. Initial COM should be full frame COM
    com1 = com_sparse(sp, crop_to=10, version=version)
    assert round(com1[0,].mean()) == 30

    # Test crop_to input as tuple. Initial COM should be full frame COM
    com1 = com_sparse(sp, crop_to=(10, 5), version=version)
    assert round(com1[0,].mean()) == 30

    # Test crop_to and init_center input.
    # No counts will be in the center so all positions will be np.nan
    com2 = com_sparse(sp, crop_to=10, init_center=(1, 1), version=version)
    assert np.isnan(com2[0,0,0])


def test_com_sparse_version_comparison(sparse_array_small):
    """Test that both versions of com_sparse produce identical results."""
    frame_shape = sparse_array_small.frame_shape
    frame_x, frame_y = frame_shape
    test_cases = [
        {"crop_to": None, "init_center": None, "replace_nans": True},
        {"crop_to": None, "init_center": None, "replace_nans": False},
        {
            "crop_to": frame_x // 2,
            "init_center": None,
            "replace_nans": True,
        },
        {
            "crop_to": (frame_x // 2, frame_y // 2),
            "init_center": None,
            "replace_nans": True,
        },
        {
            "crop_to": frame_x // 2,
            "init_center": (frame_x // 2, frame_y // 2),
            "replace_nans": True,
        },
        {
            "crop_to": frame_x // 2,
            "init_center": (frame_x // 2, frame_y // 2),
            "replace_nans": False,
        },
        {
            "crop_to": (frame_x // 2, frame_y // 4),
            "init_center": (frame_x // 2, frame_y // 2),
            "replace_nans": False,
        },
    ]

    for i, params in enumerate(test_cases, 1):
        result_v0 = com_sparse(sparse_array_small, version=0, **params)
        result_v1 = com_sparse(sparse_array_small, version=1, **params)

        if np.isnan(result_v0).any() or np.isnan(result_v1).any():
            nan_mask_v0 = np.isnan(result_v0)
            nan_mask_v1 = np.isnan(result_v1)
            assert np.array_equal(
                nan_mask_v0, nan_mask_v1
            ), f"NaN patterns don't match for case {i}"

            # For non-NaN values, they should be equal
            valid_mask = ~nan_mask_v0
            if valid_mask.any():
                np.testing.assert_allclose(
                    result_v0[valid_mask],
                    result_v1[valid_mask],
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"Non-NaN values don't match for case {i}",
                )
        else:
            # No NaN values, direct comparison
            np.testing.assert_allclose(
                result_v0,
                result_v1,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Results don't match for case {i}",
            )
