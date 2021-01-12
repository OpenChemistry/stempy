import numpy as np

from functools import reduce, wraps
import inspect


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
        expanded = self[:]
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


class SparseArray:
    def __init__(self, data, scan_shape, frame_shape, dtype=np.uint8,
                 allow_full_expand=False):
        self.data = data.ravel()
        self.scan_shape = tuple(scan_shape)
        self.frame_shape = tuple(frame_shape)
        self.dtype = dtype
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
        if len(shape) == 3 and axis == (0,):
            return True

        return len(shape) == 4 and tuple(sorted(axis)) == (0, 1)

    @arithmethic_decorators
    def max(self, axis=None, **kwargs):
        if self.is_scan_axes(axis):
            ret = np.zeros(self.frame_shape_flat, dtype=self.dtype)
            for sparse_frame in self.data:
                ret[sparse_frame] = 1
            return ret.reshape(self.frame_shape)

    @arithmethic_decorators
    def min(self, axis=None, **kwargs):
        if self.is_scan_axes(axis):
            ret = np.zeros(self.frame_shape_flat, dtype=self.dtype)
            ret[reduce(np.intersect1d, self.data)] = 1
            return ret.reshape(self.frame_shape)

    @arithmethic_decorators
    def sum(self, axis=None, dtype=None, **kwargs):
        if dtype is None:
            dtype = self.dtype

        if self.is_scan_axes(axis):
            ret = np.zeros(self.frame_shape_flat, dtype=dtype)
            for sparse_frame in self.data:
                ret[sparse_frame] += 1
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

    def sparse_frame(self, indices):
        if not isinstance(indices, (list, tuple)):
            indices = (indices,)

        if len(indices) != len(self.scan_shape):
            msg = '{indices} shape does not match scan shape {self.scan_shape}'
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
        none_slice = slice(None)
        while len(key) < len(self.shape):
            key += [none_slice]

        # Convert all to slices
        non_slice_indices = ()
        for i, item in enumerate(key):
            if not isinstance(item, slice):
                non_slice_indices += (i,)
                key[i] = slice(item, item + 1)

        if all(x == none_slice for x in key) and not self.allow_full_expand:
            raise Exception('Full expansion is not allowed')

        data_shape = self.shape

        def slice_range(ind):
            # Get the range generated by this slice
            return range(*key[ind].indices(data_shape[ind]))

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
                    output = output[tuple(key[-expand_num:])]
                    result[tuple(result_indices)] = output
                else:
                    iterate()
                result_indices[-1] += 1
                current_indices.pop()
            result_indices.pop()

        iterate()

        # Squeeze out the non-slice indices
        return result.squeeze(axis=non_slice_indices)

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
            dp[sparse_frame] = 1
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
            dp[clipped_adjusted_sparse_frame] = 1
            return dp
        elif len(indices) == len(self.scan_shape) + 2:
            # The last two indices are frame indices
            frame_indices = indices[-2:]
            flat_frame_index = (frame_indices[0] * self.frame_shape[1] +
                                frame_indices[1])
            return 1 if flat_frame_index in sparse_frame else 0

        max_length = len(self.shape) + 1
        raise ValueError(f'0 < len(indices) < {max_length} is required')
