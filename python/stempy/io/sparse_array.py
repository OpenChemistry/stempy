import copy
from functools import wraps
import inspect

import numpy as np


def format_axis(func):
    @wraps(func)
    def wrapper(self, axis, *args, **kwargs):
        if axis is None:
            axis = self.default_axis
        elif not isinstance(axis, (list, tuple)):
            axis = (axis,)
        axis = tuple(sorted(axis))
        return func(self, axis, *args, **kwargs)
    return wrapper


def default_full_expansion(func):
    name = func.__name__

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        if ret is not None:
            return ret

        print(f'Warning: performing full expansion for {name}')
        prev_sparse_slicing = self.sparse_slicing
        self.sparse_slicing = False
        try:
            expanded = self[:]
        finally:
            self.sparse_slicing = prev_sparse_slicing
        return getattr(expanded, name)(*args, **kwargs)
    return wrapper


def warn_unimplemented_kwargs(func):
    name = func.__name__
    signature_args = inspect.getfullargspec(func)[0]

    @wraps(func)
    def wrapper(*args, **kwargs):
        for key, value in kwargs.items():
            if key not in signature_args and value is not None:
                print(f"{name}: warning - '{key}' is not implemented")

        return func(*args, **kwargs)
    return wrapper


def arithmethic_decorators(func):
    decorators = [
        format_axis,
        default_full_expansion,
        warn_unimplemented_kwargs,
    ]
    for decorate in reversed(decorators):
        func = decorate(func)

    return func


NONE_SLICE = slice(None)


class SparseArray:
    def __init__(self, data, scan_shape, frame_shape, dtype=np.uint32,
                 sparse_slicing=True, allow_full_expand=False):
        self.data = data.ravel()
        self.scan_shape = tuple(scan_shape)
        self.frame_shape = tuple(frame_shape)
        self.dtype = dtype
        self.sparse_slicing = sparse_slicing
        self.allow_full_expand = allow_full_expand

    @property
    def shape(self):
        return tuple(self.scan_shape + self.frame_shape)

    @shape.setter
    def shape(self, shape):
        if not isinstance(shape, (list, tuple)):
            shape = (shape,)

        self.reshape(*shape)

    def reshape(self, *shape):
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

    @property
    def frame_shape_flat(self):
        return (np.prod(self.frame_shape),)

    @property
    def default_axis(self):
        return tuple(np.arange(len(self.shape)))

    def is_scan_axes(self, axis):
        shape = self.shape
        if len(shape) == 3 and tuple(axis) == (0,):
            return True

        return len(shape) == 4 and tuple(sorted(axis)) == (0, 1)

    @arithmethic_decorators
    def max(self, axis=None, dtype=None, **kwargs):
        if dtype is None:
            dtype = self.dtype

        if self.is_scan_axes(axis):
            ret = np.zeros(self.frame_shape_flat, dtype=dtype)
            for sparse_frame in self.data:
                unique, counts = np.unique(sparse_frame, return_counts=True)
                ret[unique] = np.maximum(ret[unique], counts)
            return ret.reshape(self.frame_shape)

    @arithmethic_decorators
    def min(self, axis=None, dtype=None, **kwargs):
        if dtype is None:
            dtype = self.dtype

        if self.is_scan_axes(axis):
            ret = np.full(self.frame_shape_flat, np.iinfo(dtype).max, dtype)
            expanded = np.empty(self.frame_shape_flat, dtype)
            for sparse_frame in self.data:
                expanded[:] = 0
                unique, counts = np.unique(sparse_frame, return_counts=True)
                expanded[unique] = counts
                ret[:] = np.minimum(ret, expanded)
            return ret.reshape(self.frame_shape)

    @arithmethic_decorators
    def sum(self, axis=None, dtype=None, **kwargs):
        if dtype is None:
            dtype = self.dtype

        if self.is_scan_axes(axis):
            ret = np.zeros(self.frame_shape_flat, dtype=dtype)
            for sparse_frame in self.data:
                unique, counts = np.unique(sparse_frame, return_counts=True)
                ret[unique] += counts.astype(dtype)
            return ret.reshape(self.frame_shape)

    @arithmethic_decorators
    def mean(self, axis=None, dtype=None, **kwargs):
        if dtype is None:
            dtype = np.float32

        mean_length = np.prod([self.shape[x] for x in axis])
        if self.is_scan_axes(axis):
            summed = self.sum(axis, dtype=dtype)
            summed[:] /= mean_length
            return summed

    def bin_scans(self, bin_factor, in_place=False):
        if not all(x % bin_factor == 0 for x in self.scan_shape):
            raise ValueError(f'scan_shape must be equally divisible by '
                             f'bin_factor {bin_factor}')

        shape_size = len(self.scan_shape)
        new_scan_shape = tuple(x // bin_factor for x in self.scan_shape)
        flat_new_scan_shape = np.prod(new_scan_shape),
        new_data = np.empty(flat_new_scan_shape, dtype=object)

        original_reshaped = self.data.reshape(flat_new_scan_shape[0],
                                              bin_factor * shape_size)

        for i in range(original_reshaped.shape[0]):
            new_data[i] = np.concatenate(original_reshaped[i])

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

    def sparse_frame(self, indices):
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
                non_slice_indices += (i,)
                key[i] = slice(item, item + 1)

        scan_indices = np.arange(len(self.scan_shape))
        is_single_frame = all(x in non_slice_indices for x in scan_indices)

        kwargs = {
            'slices': key
        }

        if is_single_frame or not self.sparse_slicing:
            f = self.slice_dense
            kwargs['non_slice_indices'] = non_slice_indices
        else:
            f = self.slice_sparse

        return f(**kwargs)

    def slice_dense(self, slices, non_slice_indices=None):
        # non_slice_indices indicate which indices should be squeezed
        # out of the result.
        if non_slice_indices is None:
            non_slice_indices = []

        if all(x == NONE_SLICE for x in slices) and not self.allow_full_expand:
            raise Exception('Full expansion is not allowed')

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
                    output = self.expand(current_indices)
                    # This could be faster if we only expand what we need
                    output = output[tuple(slices[-expand_num:])]
                    result[tuple(result_indices)] = output
                else:
                    iterate()
                result_indices[-1] += 1
                current_indices.pop()
            result_indices.pop()

        iterate()

        # Squeeze out the non-slice indices
        return result.squeeze(axis=non_slice_indices)

    def slice_sparse(self, slices):
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
            for s, length in zip(scan_slices, self.scan_shape):
                num_items = len(slice_range(s, length))
                if num_items > 1:
                    new_scan_shape += (num_items,)
            shaped_data = self.data.reshape(self.scan_shape)
            new_frames = shaped_data[scan_slices].ravel()
        else:
            new_scan_shape = self.scan_shape
            new_frames = self.data

        new_flat_scan_shape = (np.prod(new_scan_shape),)

        if frame_shape_modified:
            new_frame_shape = ()
            for s, length in zip(frame_slices, self.frame_shape):
                new_frame_shape += (len(slice_range(s, length)),)

            # Map old frame indices to new ones. Invalid values will be -1.
            frame_indices = np.arange(self.frame_shape_flat[0]).reshape(
                self.frame_shape)
            valid_flat_frame_indices = frame_indices[frame_slices].ravel()
            new_frame_indices_map = np.full(self.frame_shape_flat, -1)

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

    def expand(self, indices):
        if not isinstance(indices, (list, tuple)):
            indices = (indices,)

        if len(indices) >= len(self.scan_shape):
            scan_indices = indices[:len(self.scan_shape)]
            sparse_frame = self.sparse_frame(scan_indices)

        if len(indices) < len(self.scan_shape):
            # len(indices) should be 1 and len(scan_shape) should be 2
            # Expand all the frames this represents
            dp = np.zeros(self.shape[1:], dtype=self.dtype)
            for i in range(self.scan_shape[1]):
                dp[i] = self[indices[0], i]
            return dp
        elif len(indices) == len(self.scan_shape):
            # Expand the frame this represents
            dp = np.zeros(self.frame_shape_flat, dtype=self.dtype)
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
