import pytest

import numpy as np

from stempy.image import com_dense, com_sparse, electron_count_frame, radial_sum_sparse
from stempy.io.sparse_array import SparseArray


def test_com_sparse(sparse_array_small, full_array_small):
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

    com = com_sparse(array)

    assert np.array_equal(com[:, 0], [[0.5], [0.5]])
    assert np.array_equal(com[:, 1], [[0.25], [0.25]])

    # Now, test and make sure the center of mass computed for some real
    # data matches the dense array version.

    com = com_sparse(sparse_array_small)

    full_com = np.empty((2, *full_array_small.shape[:2]), dtype=np.float32)
    for i in range(full_array_small.shape[0]):
        for j in range(full_array_small.shape[1]):
            full_com[:, i, j] = com_dense(full_array_small[i, j])[:, 0]

    assert np.array_equal(com, full_com)


def test_radial_sum_sparse(sparse_array_10x10):

    rr = radial_sum_sparse(sparse_array_10x10, center=(5, 5))

    assert np.array_equal(rr[0, 0, :], [0, 6, 0, 0, 3])

def test_com_sparse_parameters(simulate_sparse_array):
    
    sp = simulate_sparse_array #((100,100), (100,100), (30,70), (0.8), (10))
    
    # Test no inputs. This should be the full frame COM
    com0 = com_sparse(sp)
    assert round(com0[0,].mean()) == 30
    
    # Test crop_to input. Initial COM should be full frame COM
    com1 = com_sparse(sp, crop_to=(10,10))
    assert round(com1[0,].mean()) == 30
    
    # Test crop_to and init_center input.
    # No counts will be in the center so all positions will be np.nan
    com2 = com_sparse(sp, crop_to=(10,10), init_center=(1,1))
    assert np.isnan(com2[0,0,0])


def test_electron_count_frame():
    # Create a synthetic 2D numpy array (frame)
    frame = np.array(
        [
            [2000, 0, 1000, 0, 0],
            [0, 0, 0, 200, 0],
            [0, 0, 1000, 0, 0],
            [0, 200, 0, 200, 0],
            [0, 0, 1000, 0, 0],
        ],
        dtype=np.uint16,
    )

    dark = np.ones_like(frame) * 100

    # Define expected electron hits (coordinates)
    expected_hits = np.array(
        [
            [
                [
                    [1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                ]
            ]
        ]
    )

    # Electron
    electron_hits = electron_count_frame(
        frame, xray_threshold=10000, background_threshold=1, darkreference=dark
    )
    assert np.array_equal(
        electron_hits.to_dense(), expected_hits
    ), f"Expected {expected_hits}, but got {electron_hits}"

    # Test with no dark reference
    electron_hits = electron_count_frame(
        frame, xray_threshold=10000, background_threshold=1
    )

    # Test where dark reference removes some points
    dark = np.ones_like(frame) * 1000
    electron_hits = electron_count_frame(
        frame, xray_threshold=10000, background_threshold=1, darkreference=dark
    )
    expected_hits = np.array(
        [
            [
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ]
        ]
    )

    assert np.array_equal(
        electron_hits.to_dense(), expected_hits
    ), f"Expected {expected_hits}, but got {electron_hits}"
