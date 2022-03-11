import numpy as np
from scipy.ndimage import center_of_mass

from stempy.image import com_sparse
from stempy.io.sparse_array import SparseArray


def test_com_sparse(sparse_array_small, full_array_small):
    # Do a basic test of com_sparse() with multiple frames per scan
    # position.
    data = np.empty(4, dtype=object)
    data[0] = np.array([0])
    data[1] = np.array([3])
    data[2] = np.array([0])
    data[3] = np.array([0, 1, 2])
    kwargs = {
        'data': data,
        'scan_shape': (2, 1),
        'frame_shape': (2, 2),
        'scan_positions': (0, 0, 1, 1),
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
            # Need to swap com order, as com_sparse() does
            full_com[:, i, j] = center_of_mass(full_array_small[i, j])[::-1]

    assert np.array_equal(com, full_com)
