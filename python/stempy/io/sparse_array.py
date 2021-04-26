import copy
from functools import wraps
import inspect
import sys

import numpy as np


def _format_axis(func):
    @wraps(func)
    def wrapper(self, axis, *args, **kwargs):
        if axis is None:
            axis = self._default_axis
        elif not isinstance(axis, (list, tuple)):
            axis = (axis,)
        axis = tuple(sorted(axis))
        return func(self, axis, *args, **kwargs)
    return wrapper


def _default_full_expansion(func):
    name = func.__name__

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        if ret is not None:
            return ret

        _warning(f'performing full expansion for {name}')
        prev_sparse_slicing = self.sparse_slicing
        self.sparse_slicing = False
        try:
            expanded = self[:]
        finally:
            self.sparse_slicing = prev_sparse_slicing
        return getattr(expanded, name)(*args, **kwargs)
    return wrapper


def _warn_unimplemented_kwargs(func):
    name = func.__name__
    signature_args = inspect.getfullargspec(func)[0]

    @wraps(func)
    def wrapper(*args, **kwargs):
        for key, value in kwargs.items():
            if key not in signature_args and value is not None:
                _warning(f'"{key}" is not implemented for {name}')

        return func(*args, **kwargs)
    return wrapper


def _arithmethic_decorators(func):
    decorators = [
        _format_axis,
        _default_full_expansion,
        _warn_unimplemented_kwargs,
    ]
    for decorate in reversed(decorators):
        func = decorate(func)

    return func


NONE_SLICE = slice(None)


class SparseArray:
    """Utility class for analyzing sparse electron counted data

    This implements functions performed directly on the sparse data
    including binning scans/frames, and min/max/sum/mean operations
    for certain axes.

    Standard slicing notation is also supported, including start, stop,
    and stride. Slicing returns either a new sparse array or a dense
    array depending on the user-defined settings.
    """
    def __init__(self, data, scan_shape, frame_shape, dtype=np.uint32,
                 sparse_slicing=True, allow_full_expand=False):
        """Initialize a sparse array.

        :param data: the sparse array data, where the arrays represent
                     individual frames, and each value in the arrays
                     represent 1 at that index.
        :type data: np.ndarray (1D) of np.ndarray (1D)
        :param scan_shape: the shape of the scan
        :type scan_shape: tuple of length 1 or 2
        :param frame_shape: the shape of the frame
        :type frame_shape: tuple of length 2
        :param dtype: the dtype to use for expansions and arithmetic
        :type dtype: np.dtype
        :param sparse_slicing: when slicing, return another sparse
                               array, except when an individual frame
                               is returned. If False, a dense array
                               will be returned.
        :type sparse_slicing: bool
        :param allow_full_expand: allow full expansions of the data.
                                  This parameter is used to prevent
                                  full expansions of the data when it
                                  would take up too much memory. If
                                  False and a full expansion is
                                  attempted anywhere, an exception will
                                  be raised.
        :type allow_full_expand: bool
        """
        self.data = data.ravel()
        self.scan_shape = tuple(scan_shape)
        self.frame_shape = tuple(frame_shape)
        self.dtype = dtype
        self.sparse_slicing = sparse_slicing
        self.allow_full_expand = allow_full_expand

    @classmethod
    def from_electron_counts(cls, electron_counts, **init_kwargs):
        """Create a SparseArray from an ElectronCountedData namedtuple

        :param electron_counts: electron counted data (usually obtained
                                from `stempy.io.load_electron_counts()`)
        :type electron_counts: ElectronCountedData namedtuple
        :param init_kwargs: any kwargs to forward to SparseArray.__init__()
        :type init_kwargs: dict

        :return: the generated sparse array
        :rtype: SparseArray
        """
        kwargs = {
            'data': electron_counts.data,
            'scan_shape': electron_counts.scan_dimensions[::-1],
            'frame_shape': electron_counts.frame_dimensions,
        }
        kwargs.update(init_kwargs)
        return cls(**kwargs)

    @classmethod
    def from_hdf5(cls, filepath, **init_kwargs):
        """Create a SparseArray from a stempy HDF5 file

        :param filepath: the path to the HDF5 file
        :type filepath: str
        :param init_kwargs: any kwargs to forward to SparseArray.__init__()
        :type init_kwargs: dict

        :return: the generated sparse array
        :rtype: SparseArray
        """

        # This is here instead of at the top so the basic SparseArray does
        # not depend on the stempy C++ python modules being present.
        from stempy.io import load_electron_counts

        electron_counts = load_electron_counts(filepath)
        return cls.from_electron_counts(electron_counts, **init_kwargs)

    @property
    def shape(self):
        """The full shape of the data (scan shape + frame shape)

        :return: the full shape of the data
        :rtype: tuple of length 3 or 4
        """
        return tuple(self.scan_shape + self.frame_shape)

    @shape.setter
    def shape(self, shape):
        """Set the shape of the data (scan shape + frame shape)

        :param shape: the new shape of the data.
        :type shape: tuple of length 3 or 4
        """
        if not isinstance(shape, (list, tuple)):
            shape = (shape,)

        self.reshape(*shape)

    @property
    def ndim(self):
        """The number of dimensions of the data

        This is equal to len(scan shape + frame shape)

        :return: the number of dimensions of the data
        :rtype: int
        """
        return len(self.shape)

    def reshape(self, *shape):
        """Set the shape of the data (scan shape + frame shape)

        :param shape: the new shape of the data.
        :type shape: argument list or tuple of length 3 or 4

        :return: self
        :rtype: SparseArray
        """
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        if len(shape) not in (3, 4):
            raise ValueError('Shape must be length 3 or 4')

        scan_shape = shape[:-2]
        frame_shape = shape[-2:]

        # Make sure the shapes are valid
        if np.prod(scan_shape) != np.prod(self.scan_shape):
            raise ValueError('Cannot reshape scan array of size '
                             f'{self.scan_shape} into shape {scan_shape}')

        if np.prod(frame_shape) != np.prod(self.frame_shape):
            raise ValueError('Cannot reshape frame array of size '
                             f'{self.frame_shape} into shape {frame_shape}')

        # If we made it here, we can perform the reshaping...
        self.scan_shape = scan_shape
        self.frame_shape = frame_shape

        return self

    def ravel_scans(self):
        """Reshape the SparseArray so the scan shape is flattened

        The resulting SparseArray will be 3D, with 1 scan dimension
        and 2 frame dimensions.

        :return: self
        :rtype: SparseArray
        """
        self.shape = (np.prod(self.scan_shape), *self.frame_shape)
        return self

    @_arithmethic_decorators
    def max(self, axis=None, dtype=None, **kwargs):
        """Return the maximum along a given axis.

        For specialized axes, a quick, optimized max will be performed
        using the sparse data. For non specialized axes, a full
        expansion will be performed and then the max taken of that.

        Current specialized axes are:
        1. (0,) for 3D shape
        2. (0, 1) for 4D shape

        :param axis: the axis along which to perform the max
        :type axis: int or tuple
        :param dtype: the type of array to create and return.
                      Defaults to self.dtype.
        :type dtype: np.dtype
        :param kwargs: any kwargs to pass to a non specialized axes
                       operation
        :type kwargs: dict

        :return: the result of the max
        :rtype: np.ndarray
        """
        if dtype is None:
            dtype = self.dtype

        if self._is_scan_axes(axis):
            ret = np.zeros(self._frame_shape_flat, dtype=dtype)
            for sparse_frame in self.data:
                unique, counts = np.unique(sparse_frame, return_counts=True)
                ret[unique] = np.maximum(ret[unique], counts)
            return ret.reshape(self.frame_shape)

    @_arithmethic_decorators
    def min(self, axis=None, dtype=None, **kwargs):
        """Return the minimum along a given axis.

        For specialized axes, a quick, optimized min will be performed
        using the sparse data. For non specialized axes, a full
        expansion will be performed and then the min taken of that.

        Current specialized axes are:
        1. (0,) for 3D shape
        2. (0, 1) for 4D shape

        :param axis: the axis along which to perform the min
        :type axis: int or tuple
        :param dtype: the type of array to create and return.
                      Defaults to self.dtype.
        :type dtype: np.dtype
        :param kwargs: any kwargs to pass to a non specialized axes
                       operation
        :type kwargs: dict

        :return: the result of the min
        :rtype: np.ndarray
        """
        if dtype is None:
            dtype = self.dtype

        if self._is_scan_axes(axis):
            ret = np.full(self._frame_shape_flat, np.iinfo(dtype).max, dtype)
            expanded = np.empty(self._frame_shape_flat, dtype)
            for sparse_frame in self.data:
                expanded[:] = 0
                unique, counts = np.unique(sparse_frame, return_counts=True)
                expanded[unique] = counts
                ret[:] = np.minimum(ret, expanded)
            return ret.reshape(self.frame_shape)

    @_arithmethic_decorators
    def sum(self, axis=None, dtype=None, **kwargs):
        """Return the sum along a given axis.

        For specialized axes, a quick, optimized sum will be performed
        using the sparse data. For non specialized axes, a full
        expansion will be performed and then the sum taken of that.

        Current specialized axes are:
        1. (0,) for 3D shape
        2. (0, 1) for 4D shape

        :param axis: the axis along which to perform the sum
        :type axis: int or tuple
        :param dtype: the type of array to create and return.
                      Defaults to self.dtype.
        :type dtype: np.dtype
        :param kwargs: any kwargs to pass to a non specialized axes
                       operation
        :type kwargs: dict

        :return: the result of the sum
        :rtype: np.ndarray
        """
        if dtype is None:
            dtype = self.dtype

        if self._is_scan_axes(axis):
            ret = np.zeros(self._frame_shape_flat, dtype=dtype)
            for sparse_frame in self.data:
                unique, counts = np.unique(sparse_frame, return_counts=True)
                ret[unique] += counts.astype(dtype)
            return ret.reshape(self.frame_shape)

    @_arithmethic_decorators
    def mean(self, axis=None, dtype=None, **kwargs):
        """Return the mean along a given axis.

        For specialized axes, a quick, optimized mean will be performed
        using the sparse data. For non specialized axes, a full
        expansion will be performed and then the mean taken of that.

        Current specialized axes are:
        1. (0,) for 3D shape
        2. (0, 1) for 4D shape

        :param axis: the axis along which to perform the mean
        :type axis: int or tuple
        :param dtype: the type of array to create and return.
                      Defaults to np.float32.
        :type dtype: np.dtype
        :param kwargs: any kwargs to pass to a non specialized axes
                       operation
        :type kwargs: dict

        :return: the result of the mean
        :rtype: np.ndarray
        """
        if dtype is None:
            dtype = np.float32

        mean_length = np.prod([self.shape[x] for x in axis])
        if self._is_scan_axes(axis):
            summed = self.sum(axis, dtype=dtype)
            summed[:] /= mean_length
            return summed

    def bin_scans(self, bin_factor, in_place=False):
        """Perform a binning on the scan dimensions

        This will sum sparse frames together to reduce the scan
        dimensions. The scan dimensions are the first 1 or 2 dimension.

        :param bin_factor: the factor to use for binning
        :type bin_factor: int
        :param in_place: whether to modify the current SparseArray or
                         create and return a new one.
        :type in_place: bool

        :return: self if in_place is True, otherwise a new SparseArray
        :rtype: SparseArray
        """
        if not all(x % bin_factor == 0 for x in self.scan_shape):
            raise ValueError(f'scan_shape must be equally divisible by '
                             f'bin_factor {bin_factor}')

        new_scan_shape = tuple(x // bin_factor for x in self.scan_shape)
        flat_new_scan_shape = np.prod(new_scan_shape),
        new_data = np.empty(flat_new_scan_shape, dtype=object)

        original_reshaped = self.data.reshape(new_scan_shape[0], bin_factor,
                                              new_scan_shape[1], bin_factor)

        for i in range(new_scan_shape[0]):
            for j in range(new_scan_shape[1]):
                idx = i * new_scan_shape[1] + j
                to_combine = original_reshaped[i, :, j, :].ravel()
                new_data[idx] = np.concatenate(to_combine)

        if in_place:
            self.data = new_data
            self.scan_shape = new_scan_shape
            return self

        kwargs = {
            'data': new_data,
            'scan_shape': new_scan_shape,
            'frame_shape': self.frame_shape,
            'dtype': self.dtype,
            'sparse_slicing': self.sparse_slicing,
            'allow_full_expand': self.allow_full_expand,
        }
        return SparseArray(**kwargs)

    def bin_frames(self, bin_factor, in_place=False):
        """Perform a binning on the frame dimensions

        This will sum frame values together to reduce the frame
        dimensions. The frame dimensions are the last 2 dimensions.

        :param bin_factor: the factor to use for binning
        :type bin_factor: int
        :param in_place: whether to modify the current SparseArray or
                         create and return a new one.
        :type in_place: bool

        :return: self if in_place is True, otherwise a new SparseArray
        :rtype: SparseArray
        """
        if not all(x % bin_factor == 0 for x in self.frame_shape):
            raise ValueError(f'frame_shape must be equally divisible by '
                             f'bin_factor {bin_factor}')

        rows = self.data // self.frame_shape[0] // bin_factor
        cols = self.data % self.frame_shape[1] // bin_factor

        rows *= (self.frame_shape[0] // bin_factor)
        rows += cols

        new_frame_shape = tuple(x // bin_factor for x in self.frame_shape)
        if in_place:
            self.data = rows
            self.frame_shape = new_frame_shape
            return self

        kwargs = {
            'data': rows,
            'scan_shape': self.scan_shape,
            'frame_shape': new_frame_shape,
            'dtype': self.dtype,
            'sparse_slicing': self.sparse_slicing,
            'allow_full_expand': self.allow_full_expand,
        }
        return SparseArray(**kwargs)

    def __getitem__(self, key):
        # Make sure it is a list
        if not isinstance(key, (list, tuple)):
            key = [key]
        else:
            key = list(key)

        # Add any missing slices
        while len(key) < len(self.shape):
            key += [NONE_SLICE]

        # Convert all to slices
        non_slice_indices = ()
        for i, item in enumerate(key):
            if not isinstance(item, slice):
                if item >= self.shape[i] or item < -self.shape[i]:
                    raise IndexError(f'index {item} is out of bounds for '
                                     f'axis {i} with size {self.shape[i]}')

                non_slice_indices += (i,)
                if item == -1:
                    # slice(-1, 0) will not work, since negative and
                    # positive numbers are treated differently.
                    # Instead, set the next number to the last number.
                    next_num = self.shape[i]
                else:
                    next_num = item + 1

                key[i] = slice(item, next_num)

        scan_indices = np.arange(len(self.scan_shape))
        is_single_frame = all(x in non_slice_indices for x in scan_indices)

        kwargs = {
            'slices': key,
            'non_slice_indices': non_slice_indices,
        }

        if is_single_frame or not self.sparse_slicing:
            f = self._slice_dense
        else:
            f = self._slice_sparse

        return f(**kwargs)

    @property
    def _frame_shape_flat(self):
        return (np.prod(self.frame_shape),)

    @property
    def _default_axis(self):
        return tuple(np.arange(len(self.shape)))

    def _is_scan_axes(self, axis):
        shape = self.shape
        if len(shape) == 3 and tuple(axis) == (0,):
            return True

        return len(shape) == 4 and tuple(sorted(axis)) == (0, 1)

    def _sparse_frame(self, indices):
        if not isinstance(indices, (list, tuple)):
            indices = (indices,)

        if len(indices) != len(self.scan_shape):
            msg = (f'{indices} shape does not match scan shape'
                   f'{self.scan_shape}')
            raise ValueError(msg)

        if len(indices) == 1:
            scan_ind = indices[0]
        else:
            scan_ind = indices[0] * self.scan_shape[1] + indices[1]

        return self.data[scan_ind]

    def _slice_dense(self, slices, non_slice_indices=None):
        # non_slice_indices indicate which indices should be squeezed
        # out of the result.
        if non_slice_indices is None:
            non_slice_indices = []

        if all(x == NONE_SLICE for x in slices) and not self.allow_full_expand:
            raise FullExpansionDenied('Full expansion is not allowed')

        data_shape = self.shape

        def slice_range(ind):
            # Get the range generated by this slice
            return range(*slices[ind].indices(data_shape[ind]))

        # Determine the shape of the result
        result_shape = ()
        for i in range(len(data_shape)):
            result_shape += (len(slice_range(i)),)

        # Create the result
        result = np.zeros(result_shape, dtype=self.dtype)

        # We will currently expand whole frames at a time
        expand_num = len(self.scan_shape)

        # Lists to use in the recursion
        current_indices = []
        result_indices = []

        def iterate():
            ind = len(current_indices)
            result_indices.append(0)
            for i in slice_range(ind):
                current_indices.append(i)
                if len(current_indices) == expand_num:
                    output = self._expand(current_indices)
                    # This could be faster if we only expand what we need
                    output = output[tuple(slices[-output.ndim:])]
                    result[tuple(result_indices)] = output
                else:
                    iterate()
                result_indices[-1] += 1
                current_indices.pop()
            result_indices.pop()

        iterate()

        # Squeeze out the non-slice indices
        return result.squeeze(axis=non_slice_indices)

    def _slice_sparse(self, slices, non_slice_indices=None):
        # non_slice_indices indicate which indices should be squeezed
        # out of the result.
        if non_slice_indices is None:
            non_slice_indices = []

        if len(slices) != len(self.shape):
            raise Exception('Slices must be same length as shape')

        if any(not isinstance(x, slice) for x in slices):
            raise Exception('All slices must be slice objects')

        scan_slices = tuple(slices[:len(self.scan_shape)])
        frame_slices = tuple(slices[len(self.scan_shape):])

        scan_shape_modified = any(x != NONE_SLICE for x in scan_slices)
        frame_shape_modified = any(x != NONE_SLICE for x in frame_slices)

        def slice_range(slice, length):
            # Get the range generated by this slice
            return range(*slice.indices(length))

        if scan_shape_modified:
            new_scan_shape = ()
            for i, (s, length) in enumerate(zip(scan_slices, self.scan_shape)):
                if i not in non_slice_indices:
                    new_scan_shape += (len(slice_range(s, length)),)
            shaped_data = self.data.reshape(self.scan_shape)
            new_frames = shaped_data[scan_slices].ravel()
        else:
            new_scan_shape = self.scan_shape
            new_frames = self.data

        new_flat_scan_shape = (np.prod(new_scan_shape),)

        if frame_shape_modified:
            new_frame_shape = ()
            for i, (s, length) in enumerate(zip(frame_slices,
                                                self.frame_shape)):
                if i + len(self.scan_shape) not in non_slice_indices:
                    new_frame_shape += (len(slice_range(s, length)),)

            # Map old frame indices to new ones. Invalid values will be -1.
            frame_indices = np.arange(self._frame_shape_flat[0]).reshape(
                self.frame_shape)
            valid_flat_frame_indices = frame_indices[frame_slices].ravel()
            new_frame_indices_map = np.full(self._frame_shape_flat, -1)

            new_indices = np.arange(len(valid_flat_frame_indices))
            new_frame_indices_map[valid_flat_frame_indices] = new_indices

            # Allocate the new data
            new_data = np.empty(new_flat_scan_shape, dtype=object)

            # Now set it
            for i, frame in enumerate(new_frames):
                new_frame = new_frame_indices_map[frame]
                new_data[i] = copy.deepcopy(new_frame[new_frame >= 0])
        else:
            new_frame_shape = self.frame_shape
            new_data = copy.deepcopy(new_frames)

        kwargs = {
            'data': new_data,
            'scan_shape': new_scan_shape,
            'frame_shape': new_frame_shape,
            'dtype': self.dtype,
            'sparse_slicing': self.sparse_slicing,
            'allow_full_expand': self.allow_full_expand,
        }
        return SparseArray(**kwargs)

    def _expand(self, indices):
        if not isinstance(indices, (list, tuple)):
            indices = (indices,)

        if len(indices) >= len(self.scan_shape):
            scan_indices = indices[:len(self.scan_shape)]
            sparse_frame = self._sparse_frame(scan_indices)

        if len(indices) < len(self.scan_shape):
            # len(indices) should be 1 and len(scan_shape) should be 2
            # Expand all the frames this represents
            dp = np.zeros(self.shape[1:], dtype=self.dtype)
            for i in range(self.scan_shape[1]):
                dp[i] = self[indices[0], i]
            return dp
        elif len(indices) == len(self.scan_shape):
            # Expand the frame this represents
            dp = np.zeros(self._frame_shape_flat, dtype=self.dtype)
            unique, counts = np.unique(sparse_frame, return_counts=True)
            dp[unique] += counts.astype(self.dtype)
            return dp.reshape(self.frame_shape)
        elif len(indices) == len(self.scan_shape) + 1:
            # The last index is a frame index
            frame_index = indices[-1]
            dp = np.zeros((self.frame_shape[1],), dtype=self.dtype)
            start = frame_index * self.frame_shape[1]
            end = start + self.frame_shape[1] - 1
            in_range = np.logical_and(sparse_frame >= start,
                                      sparse_frame <= end)
            clipped_adjusted_sparse_frame = sparse_frame[in_range] - start
            unique, counts = np.unique(clipped_adjusted_sparse_frame,
                                       return_counts=True)
            dp[unique] += counts.astype(self.dtype)
            return dp
        elif len(indices) == len(self.scan_shape) + 2:
            # The last two indices are frame indices
            frame_indices = indices[-2:]
            flat_frame_index = (frame_indices[0] * self.frame_shape[1] +
                                frame_indices[1])
            return np.count_nonzero(sparse_frame == flat_frame_index)

        max_length = len(self.shape) + 1
        raise ValueError(f'0 < len(indices) < {max_length} is required')


def _warning(msg):
    print(f'Warning: {msg}', file=sys.stderr)


class FullExpansionDenied(Exception):
    pass
