import numpy as np

try:
    # Use importlib metadata if available (python >=3.8)
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # No importlib metadata. Try to use pkg_resources instead.
    from pkg_resources import (
        get_distribution,
        DistributionNotFound as PackageNotFoundError,
    )

    def version(x):
        return get_distribution(x).version


def get_version():
    try:
        return version('stempy')
    except PackageNotFoundError:
        # package is not installed
        pass


def is_numpy_v2():
    # https://github.com/scipy/scipy/pull/20172/files#diff-d28fbb0be287769c763ce61b0695feb0a6ca0e67b28bafee20526b46be7aaa84R61
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        return True
    else:
        return False


def safe_array(data, copy=False):
    """Create numpy array with safe handling for numpy 2.0+ copy behavior.

    In numpy 2.0+, np.array(..., copy=False) raises an error if a copy is needed.
    This function uses np.asarray() for numpy 2.0+ and np.array(..., copy=False)
    for older versions.

    :param data: input data to convert to array
    :param copy: for numpy < 2.0, whether to avoid copying data
    :return: numpy array
    """
    if is_numpy_v2():
        # For numpy 2.0+, use asarray which allows copies when needed
        return np.asarray(data)
    else:
        # For numpy < 2.0, use the old behavior
        return np.array(data, copy=copy)
