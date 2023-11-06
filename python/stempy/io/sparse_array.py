from collections.abc import Sequence
import copy
from functools import wraps
import inspect
import sys

import numpy as np

from .compatibility import convert_data_format


def _format_axis(func):
    @wraps(func)
    def wrapper(self, axis=None, *args, **kwargs):
        if axis == 'scan':
            axis = self.scan_axes
        elif axis == 'frame':
            axis = self.frame_axes

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
    VERSION = 3

    def __init__(self, data, scan_shape, frame_shape, dtype=np.uint32,
                 sparse_slicing=True, allow_full_expand=False, metadata=None):
        """Initialize a sparse array.

        :param data: the sparse array data, where the outer array represents
                     scan position, the middle array represents a frame taken
                     at the scan position, and the inner array represents the
                     sparse frame data, where each value in the arrays
                     represent 1 at that index.
        :type data: np.ndarray (2D) of np.ndarray (1D)
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
        :param metadata: a dict containing any metadata. This will be
                         saved and loaded with the HDF5 methods. If
                         None, it will default to an empty dict.
        :type metadata: dict or None
        """
        if data.ndim == 1:
            # Give it an extra dimension for the frame index
            data = data[:, np.newaxis]

        self.data = data
        self.scan_shape = tuple(scan_shape)
        self.frame_shape = tuple(frame_shape)
        self.dtype = dtype
        self.sparse_slicing = sparse_slicing
        self.allow_full_expand = allow_full_expand

        # Prevent obscure errors later by validating now
        self._validate()

        if metadata is None:
            metadata = {}

        self.metadata = metadata

    def _validate(self):
        if self.data.ndim != 2:
            msg = (
                'The data must have 2 dimensions, but instead has '
                f'{self.data.ndim}. The outer dimension should be the '
                'scan position, and the inner dimension should be the '
                'frame index at that scan position'
            )
            raise Exception(msg)

        if len(self.data) != self.num_scans:
            msg = (
                f'The length of the data array ({len(self.data)}) must be '
                f'equal to the number of scans ({self.num_scans}), which '
                f'is computed via np.prod(scan_shape)'
            )
            raise Exception(msg)

        for i, scan_frames in enumerate(self.data):
            for j, frame in enumerate(scan_frames):
                if not np.issubdtype(frame.dtype, np.integer):
                    msg = (
                        f'All frames of data must have integral dtype, but '
                        f'scan_position={i} frame={j} has a dtype of '
                        f'{frame.dtype}'
                    )
                    raise Exception(msg)

    @classmethod
    def from_hdf5(cls, filepath, keep_flyback=True, **init_kwargs):
        """Create a SparseArray from a stempy HDF5 file

        :param filepath: the path to the HDF5 file
        :type filepath: str
        :param keep_flyback: option to crop the flyback column during loading
        :type keep_flyback: bool
        :param init_kwargs: any kwargs to forward to SparseArray.__init__()
        :type init_kwargs: dict

        :return: the generated sparse array
        :rtype: SparseArray
        """
        import h5py

        with h5py.File(filepath, 'r') as f:
            version = f.attrs.get('version', 1)

            frames = f['electron_events/frames']
            scan_positions_group = f['electron_events/scan_positions']
            scan_shape = [scan_positions_group.attrs[x] for x in ['Nx', 'Ny']]
            frame_shape = [frames.attrs[x] for x in ['Nx', 'Ny']]
            
            if keep_flyback:
                data = frames[()]
            else:
                orig_indices = np.ravel_multi_index([ii.ravel() for ii in np.indices(scan_shape)],scan_shape)
                crop_indices = np.delete(orig_indices, orig_indices[::scan_shape[1]])
                data = frames[crop_indices]
                scan_shape[1] = scan_shape[1] - 1

            
            # Load any metadata
            metadata = {}
            if 'metadata' in f:
                load_h5_to_dict(f['metadata'], metadata)

        scan_shape = scan_shape[::-1]

        if version >= 3:
            # Convert to int to avoid integer division that results in 
            # a float
            frames_per_scan = len(data) // np.prod(scan_shape, dtype=int)
            # Need to reshape the data, as it was flattened before saving
            data = data.reshape((np.prod(scan_shape), frames_per_scan))

        # We may need to convert the version of the data
        if version != cls.VERSION:
            kwargs = {
                'data': data,
                'scan_positions': scan_positions,
                'scan_shape': scan_shape,
                'frame_shape': frame_shape,
                'from_version': version,
                'to_version': cls.VERSION,
            }
            data = convert_data_format(**kwargs)

        kwargs = {
            'data': data,
            'scan_shape': scan_shape,
            'frame_shape': frame_shape,
            'metadata': metadata,
        }
        kwargs.update(init_kwargs)
        return cls(**kwargs)

    def write_to_hdf5(self, path):
        """Save the SparseArray to an HDF5 file.

        :param path: path to the HDF5 file.
        :type path: str
        """
        import h5py

        data = self.data
        with h5py.File(path, 'a') as f:
            f.attrs['version'] = self.VERSION

            group = f.require_group('electron_events')
            scan_positions = group.create_dataset('scan_positions',
                                                  data=self.scan_positions)
            scan_positions.attrs['Nx'] = self.scan_shape[1]
            scan_positions.attrs['Ny'] = self.scan_shape[0]

            # We can't store a 2D array that contains variable length arrays
            # in h5py currently. So flatten the data, and we will reshape it
            # when we read it back in. See h5py/h5py#876
            coordinates_type = h5py.special_dtype(vlen=np.uint32)
            frames = group.create_dataset('frames', (np.prod(data.shape),),
                                          dtype=coordinates_type)
            # Add the frame dimensions as attributes
            frames.attrs['Nx'] = self.frame_shape[0]
            frames.attrs['Ny'] = self.frame_shape[1]

            frames[...] = data.ravel()

            # Write out the metadata
            save_dict_to_h5(self.metadata, f.require_group('metadata'))

    def to_dense(self):
        """Create and return a fully dense version of the sparse array

        If the array shape is large, this may cause the system to
        run out of memory.

        This is equivalent to `array[:]` if `allow_full_expand` is `True`
        and `sparse_slicing` is `False`.

        :return: the fully dense array
        :rtype: np.ndarray
        """
        prev_allow_full_expand = self.allow_full_expand
        prev_sparse_slicing = self.sparse_slicing

        self.allow_full_expand = True
        self.sparse_slicing = False
        try:
            return self[:]
        finally:
            self.allow_full_expand = prev_allow_full_expand
            self.sparse_slicing = prev_sparse_slicing

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
            for scan_frames in self.data:
                concatenated = np.concatenate(scan_frames)
                unique, counts = np.unique(concatenated, return_counts=True)
                ret[unique] = np.maximum(ret[unique], counts)
            return ret.reshape(self.frame_shape)

        if self._is_frame_axes(axis):
            ret = np.zeros(self._scan_shape_flat, dtype=dtype)
            for i, scan_frames in enumerate(self.data):
                concatenated = np.concatenate(scan_frames)
                unique, counts = np.unique(concatenated, return_counts=True)
                ret[i] = np.max(counts)
            return ret.reshape(self.scan_shape)

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
            for scan_frames in self.data:
                expanded[:] = 0
                concatenated = np.concatenate(scan_frames)
                unique, counts = np.unique(concatenated, return_counts=True)
                expanded[unique] = counts
                ret[:] = np.minimum(ret, expanded)
            return ret.reshape(self.frame_shape)

        if self._is_frame_axes(axis):
            ret = np.full(self._scan_shape_flat, np.iinfo(dtype).max, dtype)
            for i, scan_frames in enumerate(self.data):
                concatenated = np.concatenate(scan_frames)
                unique, counts = np.unique(concatenated, return_counts=True)
                if len(counts) < self._frame_shape_flat[0]:
                    # Some pixels are 0. The min must be zero.
                    ret[i] = 0
                else:
                    # The min is equal to the minimum in the counts.
                    ret[i] = np.min(counts)
            return ret.reshape(self.scan_shape)

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
            for scan_frames in self.data:
                for sparse_frame in scan_frames:
                    ret[sparse_frame] += 1
            return ret.reshape(self.frame_shape)

        if self._is_frame_axes(axis):
            ret = np.zeros(self._scan_shape_flat, dtype=dtype)
            for i, scan_frames in enumerate(self.data):
                for sparse_frame in scan_frames:
                    ret[i] += len(sparse_frame)
            return ret.reshape(self.scan_shape)

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

        # If the sum is specialized along this axis, this will be fast.
        # Otherwise, the sum will perform a full expansion.
        summed = self.sum(axis, dtype=dtype)
        mean_length = np.prod([self.shape[x] for x in axis])

        if isinstance(summed, np.ndarray):
            # Divide in place to avoid a copy
            summed[:] /= mean_length
        else:
            # It's a scalar
            summed /= mean_length

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

        # No need to modify the data, just the scan shape and the scan
        # positions.
        new_scan_shape = tuple(x // bin_factor for x in self.scan_shape)
        all_positions = self.scan_positions
        all_positions_reshaped = all_positions.reshape(
            new_scan_shape[0], bin_factor, new_scan_shape[1], bin_factor)

        new_data_shape = (np.prod(new_scan_shape),
                          bin_factor**2 * self.num_frames_per_scan)
        new_data = np.empty(new_data_shape, dtype=object)
        for i in range(new_scan_shape[0]):
            for j in range(new_scan_shape[1]):
                idx = i * new_scan_shape[1] + j
                scan_position = all_positions[idx]
                to_change = all_positions_reshaped[i, :, j, :].ravel()
                new_data[scan_position] = np.concatenate(self.data[to_change])

        if in_place:
            self.scan_shape = new_scan_shape
            self.data = new_data
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

        # Ravel the data for easier manipulation in this function
        data = self.data.ravel()

        rows = data // self.frame_shape[0] // bin_factor
        cols = data % self.frame_shape[1] // bin_factor

        rows *= (self.frame_shape[0] // bin_factor)
        rows += cols

        new_data = rows
        frames_per_scan = self.data.shape[1]
        raveled_scan_positions = np.repeat(self.scan_positions,
                                           frames_per_scan)
        new_scan_positions = raveled_scan_positions

        # Place duplicates in separate frames
        extra_rows = []
        extra_scan_positions = []
        for i, row in enumerate(rows):
            unique, counts = np.unique(row, return_counts=True)
            rows[i] = unique

            # Ensure counts can be negative
            counts = counts.astype(np.int64) - 1
            while np.any(counts > 0):
                extra_rows.append(unique[counts > 0])
                extra_scan_positions.append(raveled_scan_positions[i])
                counts -= 1

        if extra_rows:
            # Resize the new data and add on the extra rows
            new_size = rows.shape[0] + len(extra_rows)
            new_data = np.empty(new_size, dtype=object)
            new_data[:rows.shape[0]] = rows
            new_data[rows.shape[0]:] = extra_rows
            new_scan_positions = np.append(new_scan_positions,
                                           extra_scan_positions)

            # Find the max number of extra frames per scan position and use
            # that.
            unique, counts = np.unique(new_scan_positions, return_counts=True)
            frames_per_scan = np.max(counts)

        # The data was raveled. Reshape it with the frames per scan.
        result = np.empty((self.num_scans, frames_per_scan), dtype=object)

        # Now set all of the data in their new locations
        for datum, pos in zip(new_data, new_scan_positions):
            current = result[pos]
            for i in range(frames_per_scan):
                if current[i] is None:
                    current[i] = datum
                    break

        # Now fill in any remaining Nones with empty arrays
        for scan_frames in result:
            for i in range(frames_per_scan):
                if scan_frames[i] is None:
                    scan_frames[i] = np.array([], dtype=np.uint32)

        new_data = result

        new_frame_shape = tuple(x // bin_factor for x in self.frame_shape)
        if in_place:
            self.data = new_data
            self.frame_shape = new_frame_shape
            return self

        kwargs = {
            'data': new_data,
            'scan_shape': self.scan_shape,
            'frame_shape': new_frame_shape,
            'dtype': self.dtype,
            'sparse_slicing': self.sparse_slicing,
            'allow_full_expand': self.allow_full_expand,
        }
        return SparseArray(**kwargs)

    def __getitem__(self, key):
        scan_slices, frame_slices = self._split_slices(key)

        if self.sparse_slicing:
            return self._slice_sparse(scan_slices, frame_slices)
        else:
            return self._slice_dense(scan_slices, frame_slices)

    @property
    def scan_positions(self):
        """Get an array of scan positions for the data

        :return: the scan positions
        :rtype: np.ndarray of int
        """
        return np.arange(self.num_scans)

    @property
    def num_scans(self):
        """Get the number of scan positions

        :return: the number of scan positions
        :rtype: int
        """
        return self._scan_shape_flat[0]

    @property
    def num_frames_per_scan(self):
        """Get the number of frames per scan position

        :return: the number of frames per scan position
        :rtype: int
        """
        return self.data.shape[1]

    @property
    def scan_axes(self):
        """Get the axes for the scan positions

        :return: the axes for the scan positions
        :rtype: tuple(int)
        """
        num = len(self.scan_shape)
        return tuple(range(num))

    @property
    def frame_axes(self):
        """Get the axes for the frame positions

        :return: the axes for the frame positions
        :rtype: tuple(int)
        """
        start = len(self.scan_shape)
        return tuple(range(start, len(self.shape)))

    @property
    def _frame_shape_flat(self):
        return (np.prod(self.frame_shape),)

    @property
    def _scan_shape_flat(self):
        return (np.prod(self.scan_shape),)

    @property
    def _default_axis(self):
        return tuple(np.arange(len(self.shape)))

    def _is_scan_axes(self, axis):
        return tuple(sorted(axis)) == self.scan_axes

    def _is_frame_axes(self, axis):
        return tuple(sorted(axis)) == self.frame_axes

    def _split_slices(self, slices):
        """Split the slices into scan slices and frame slices

        This will also perform some validation to make sure there are no
        issues.

        Returns `scan_slices, frame_slices`.
        """
        def validate_if_advanced_indexing(obj, max_ndim, is_scan):
            """Check if the obj is advanced indexing
               (and convert to ndarray if it is)

            If it is advanced indexing, ensure the number of dimensions do
            not exceed the provided max.

            returns the obj (converted to an ndarray if advanced indexing)
            """
            if isinstance(obj, Sequence):
                # If it's a sequence, it is advanced indexing.
                # Convert to ndarray.
                obj = np.asarray(obj)

            if isinstance(obj, np.ndarray):
                # If it is a numpy array, it is advanced indexing.
                # Ensure that there are not too many dimensions.
                if obj.ndim > max_ndim:
                    msg = 'Too many advanced indexing dimensions.'
                    if is_scan:
                        msg += (
                            ' Cannot perform advanced indexing on both the '
                            'scan positions and the frame positions '
                            'simultaneously'
                        )
                    raise IndexError(msg)

            return obj

        if not isinstance(slices, tuple):
            # Wrap it with a tuple to simplify
            slices = (slices,)

        if not slices:
            # It's an empty tuple
            return tuple(), tuple()

        # Figure out which slices belong to which parts.
        # The first slice should definitely be part of the scan.
        first_slice = slices[0]
        first_slice = validate_if_advanced_indexing(first_slice,
                                                    len(self.scan_shape),
                                                    is_scan=True)

        frame_start = 1
        if len(self.scan_shape) > 1:
            # We might have 2 slices for the scan shape
            if not isinstance(first_slice,
                              np.ndarray) or first_slice.ndim == 1:
                # We have another scan slice
                frame_start += 1

        if frame_start == 2 and len(slices) > 1:
            # Validate the second scan slice.
            second_slice = validate_if_advanced_indexing(slices[1], 1,
                                                         is_scan=True)
            scan_slices = (first_slice, second_slice)
        else:
            scan_slices = (first_slice,)

        # If there are frame indices, validate them too
        frame_slices = tuple()
        for i in range(frame_start, len(slices)):
            max_ndim = frame_start + 2 - i
            if max_ndim == 0:
                raise IndexError('Too many indices for frame positions')
            frame_slices += (validate_if_advanced_indexing(slices[i], max_ndim,
                                                           is_scan=False),)

        # Verify that we are not doing advanced indexing on both the scan
        # positions and frame positions simultaneously.
        if (any(isinstance(x, np.ndarray) for x in scan_slices) and
                any(isinstance(x, np.ndarray) for x in frame_slices)):
            msg = (
                'Cannot perform advanced indexing on both scan positions '
                'and frame positions simultaneously'
            )
            raise IndexError(msg)

        # Verify that if there are any 2D advanced indexing arrays, they
        # must be boolean and of the same shape.
        first_frame_slice = (slice(None) if not frame_slices
                             else frame_slices[0])
        for i, to_check in enumerate((first_slice, first_frame_slice)):
            req_shape = self.scan_shape if i == 0 else self.frame_shape
            if (isinstance(to_check, np.ndarray) and to_check.ndim == 2 and
               (to_check.dtype != np.bool_ or to_check.shape != req_shape)):
                msg = (
                    '2D advanced indexing is only allowed for boolean arrays '
                    'that match either the scan shape or the frame shape '
                    '(whichever it is indexing into)'
                )
                raise IndexError(msg)

        return scan_slices, frame_slices

    def _sparse_frames(self, indices):
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

    def _slice_dense(self, scan_slices, frame_slices):
        new_scan_shape = np.empty(self.scan_shape,
                                  dtype=bool)[scan_slices].shape
        new_frame_shape = np.empty(self.frame_shape,
                                   dtype=bool)[frame_slices].shape

        result_shape = new_scan_shape + new_frame_shape

        if result_shape == self.shape and not self.allow_full_expand:
            raise FullExpansionDenied('Full expansion is not allowed')

        # Create the result
        result = np.zeros(result_shape, dtype=self.dtype)

        all_positions = self.scan_positions.reshape(self.scan_shape)
        sliced = all_positions[scan_slices].ravel()

        scan_indices = np.unravel_index(sliced, self.scan_shape)
        scan_indices = np.array(scan_indices).T

        # We will currently expand whole frames at a time
        for i, indices in enumerate(scan_indices):
            output = self._expand(tuple(indices))
            # This could be faster if we only expand what we need
            output = output[frame_slices]
            result_indices = np.unravel_index(i, new_scan_shape)
            result[tuple(result_indices)] = output

        return result

    def _slice_sparse(self, scan_slices, frame_slices):
        scan_shape_modified = any(not _is_none_slice(x) for x in scan_slices)
        frame_shape_modified = any(not _is_none_slice(x) for x in frame_slices)

        if scan_shape_modified:
            all_positions = self.scan_positions.reshape(self.scan_shape)
            sliced = all_positions[scan_slices]

            if isinstance(sliced, np.integer):
                # Everything was sliced except one frame.
                # Return the dense frame instead.
                return self._slice_dense(scan_slices, frame_slices)

            new_scan_shape = sliced.shape
            positions_to_keep = sliced.ravel()
            new_frames = self.data[positions_to_keep]
        else:
            new_scan_shape = self.scan_shape
            new_frames = self.data

        if frame_shape_modified:

            # Map old frame indices to new ones. Invalid values will be -1.
            frame_indices = np.arange(self._frame_shape_flat[0]).reshape(
                self.frame_shape)
            sliced = frame_indices[frame_slices]
            new_frame_shape = sliced.shape

            if not new_frame_shape:
                # We don't support an empty frame shape.
                # Just return the dense array instead.
                return self._slice_dense(scan_slices, frame_slices)

            valid_flat_frame_indices = sliced.ravel()
            new_frame_indices_map = np.full(self._frame_shape_flat, -1)

            new_indices = np.arange(len(valid_flat_frame_indices))
            new_frame_indices_map[valid_flat_frame_indices] = new_indices

            # Allocate the new data
            new_data = np.empty(new_frames.shape, dtype=object)

            # Now set it
            for i, scan_frames in enumerate(new_frames):
                for j, sparse_frame in enumerate(scan_frames):
                    new_frame = new_frame_indices_map[sparse_frame]
                    new_data[i, j] = copy.deepcopy(new_frame[new_frame >= 0])
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
            sparse_frames = self._sparse_frames(scan_indices)

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
            for frame in sparse_frames:
                dp[frame] += 1
            return dp.reshape(self.frame_shape)
        elif len(indices) == len(self.scan_shape) + 1:
            # The last index is a frame index
            frame_index = indices[-1]
            dp = np.zeros((self.frame_shape[1],), dtype=self.dtype)
            start = frame_index * self.frame_shape[1]
            end = start + self.frame_shape[1] - 1
            for frame in sparse_frames:
                in_range = np.logical_and(frame >= start, frame <= end)
                clipped_adjusted_frame = frame[in_range] - start
                dp[clipped_adjusted_frame] += 1
            return dp
        elif len(indices) == len(self.scan_shape) + 2:
            # The last two indices are frame indices
            frame_indices = indices[-2:]
            flat_frame_index = (frame_indices[0] * self.frame_shape[1] +
                                frame_indices[1])
            return sum(flat_frame_index in frame for frame in sparse_frames)

        max_length = len(self.shape) + 1
        raise ValueError(f'0 < len(indices) < {max_length} is required')

    @staticmethod
    def _is_advanced_indexing(obj):
        """Look at the object to see if it is considered advanced indexing

        We will follow the logic taken from here:
        https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

        "Advanced indexing is triggered when the selection object, obj, is a
         non-tuple sequence object, an ndarray (of data type integer or bool),
         or a tuple with at least one sequence object or ndarray (of data
         type integer or bool)."
        """

        def is_int_or_bool_ndarray(x):
            """Check if x is an ndarray of type int or bool"""
            if not isinstance(x, np.ndarray):
                return False

            return issubclass(x.dtype.type, (np.integer, np.bool_))

        if not isinstance(obj, tuple) and isinstance(obj, Sequence):
            return True

        if is_int_or_bool_ndarray(obj):
            return True

        if isinstance(obj, tuple):
            return any(isinstance(x, Sequence) or is_int_or_bool_ndarray(x)
                       for x in obj)

        return False


def _is_none_slice(x):
    return isinstance(x, slice) and x == NONE_SLICE


def _warning(msg):
    print(f'Warning: {msg}', file=sys.stderr)


class FullExpansionDenied(Exception):
    pass


def save_dict_to_h5(d, group):
    for k, v in d.items():
        if isinstance(v, dict):
            new_group = group.require_group(k)
            save_dict_to_h5(v, new_group)
        else:
            group.attrs[k] = v


def load_h5_to_dict(group, d):
    import h5py

    for k, v in group.attrs.items():
        d[k] = v

    for k, v in group.items():
        if isinstance(v, h5py.Group):
            d[k] = {}
            load_h5_to_dict(v, d[k])
