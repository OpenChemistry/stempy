import numpy as np

from stempy.image import com_sparse
from stempy.io.sparse_array import SparseArray


def test_com_sparse():
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
