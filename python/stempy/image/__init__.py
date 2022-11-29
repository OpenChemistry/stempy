import warnings

from stempy import _image
from stempy import _io
from stempy.io import (
    get_hdf5_reader, ReaderMixin, PyReader, SectorThreadedReader,
    SectorThreadedMultiPassReader, SparseArray
)

import h5py
import numpy as np

def create_stem_images(input, inner_radii, outer_radii, scan_dimensions=(0, 0),
                       center=(-1, -1), frame_dimensions=None, frame_offset=0):
    """Create a series of stem images from the input.

    :param input: the file reader that has already opened the data, or an
                  open h5py file, or an ElectronCountedData namedtuple
                  containing the sparse data, or a SparseArray, or a
                  numpy.ndarray of either the sparse or the raw data (if the
                  frame_dimensions argument is supplied, numpy.ndarray is
                  inferred to be sparse data).
    :type input: stempy.io.reader, an h5py file, ElectronCountedData,
                 SparseArray, or numpy.ndarray
    :param inner_radii: a list of inner radii. Must match
                        the length of `outer_radii`.
    :type inner_radii: list of ints
    :param outer_radii: a list of outer radii. Must match
                        the length of `inner_radii`.
    :type outer_radii: list of ints
    :param scan_dimensions: the dimensions of the scan, where the order is
                            (width, height). If set to (0, 0), an attempt
                            will be made to read the scan dimensions from
                            the data file.
    :type scan_dimensions: tuple of ints of length 2
    :param center: the center of the images, where the order is (x, y). If set
                   to (-1, -1), the center will be set to
                   (scan_dimensions[0] / 2, scan_dimensions[1] / 2).
    :type center: tuple of ints of length 2
    :param frame_dimensions: the dimensions of each frame, where the order is
                             (width, height). Only used for input of type
                             numpy.ndarray, in which case its presence implies
                             that the input is sparse data rather than raw
                             data.
    :type frame_dimensions: tuple of ints of length 2
    :param frame_offset: the amount by which to offset the frame. Only used
                         for sparse data input of type numpy.ndarray.
    :type frame_offset: int

    :return: A numpy array of the STEM images.
    :rtype: numpy.ndarray
    """
    # Ensure the inner and outer radii are tuples or lists
    if not isinstance(inner_radii, (tuple, list)):
        inner_radii = [inner_radii]
    if not isinstance(outer_radii, (tuple, list)):
        outer_radii = [outer_radii]

    # Electron counted data attributes
    ecd_attrs = ['data', 'scan_dimensions', 'frame_dimensions']

    if isinstance(input, h5py._hl.files.File):
        # Handle h5py file
        input = get_hdf5_reader(input)
        imgs = _image.create_stem_images(input.begin(), input.end(),
                                         inner_radii, outer_radii,
                                         scan_dimensions, center)
    elif issubclass(type(input), ReaderMixin):
        # Handle standard reader
        imgs = _image.create_stem_images(input.begin(), input.end(),
                                         inner_radii, outer_radii,
                                         scan_dimensions, center)
    elif hasattr(input, '_electron_counted_data'):
        # Handle electron counted data with C++ object
        # This could also be a SparseArray created via electron_count()
        imgs = _image.create_stem_images(input._electron_counted_data,
                                         inner_radii, outer_radii, center)
    elif isinstance(input, SparseArray):
        imgs = _image.create_stem_images(input.data, inner_radii, outer_radii,
                                         input.scan_shape[::-1],
                                         input.frame_shape, center)
    elif all([hasattr(input, x) for x in ecd_attrs]):
        # Handle electron counted data without C++ object
        # Assume this is v1 data and that each frame is one scan position
        data = input.data
        if data.ndim == 1:
            data = data[:, np.newaxis]
        imgs = _image.create_stem_images(input.data, inner_radii, outer_radii,
                                         input.scan_dimensions,
                                         input.frame_dimensions, center)
    elif isinstance(input, np.ndarray):
        # The presence of frame dimensions implies it is sparse data
        if frame_dimensions is not None:
            # Handle sparse data
            if input.ndim == 1:
                input = input[:, np.newaxis]
            imgs = _image.create_stem_images(input, inner_radii, outer_radii,
                                             scan_dimensions, frame_dimensions,
                                             center)
        else:
            # Handle raw data
            # Make sure the scan dimensions were passed
            if not scan_dimensions or scan_dimensions == (0, 0):
                msg = ('scan_dimensions must be provided for np.ndarray '
                       'raw data input')
                raise Exception(msg)

            # Should have shape (num_images, frame_height, frame_width)
            num_images = input.shape[0]
            image_numbers = np.arange(num_images)
            block_size = 32
            reader = PyReader(input, image_numbers, scan_dimensions, block_size, num_images)

            imgs = _image.create_stem_images(reader.begin(), reader.end(),
                                             inner_radii, outer_radii,
                                             scan_dimensions, center)
    else:
        raise Exception('Type of input, ' + str(type(input)) +
                        ', is not known to stempy.image.create_stem_images()')

    images = [np.array(img, copy=False) for img in imgs]
    return np.array(images, copy=False)

def create_stem_histogram(numBins, reader, inner_radii,
                          outer_radii, scan_dimensions=(0, 0),
                          center=(-1, -1)):
    """Create a histogram of the stem images generated from the input.

    :param numBins: the number of bins the histogram should have.
    :param reader: the file reader that has already opened the data.
    :type reader: stempy.io.reader
    :param inner_radii: a list of inner radii. Must match
                        the length of `outer_radii`.
    :type inner_radii: list of ints
    :param outer_radii: a list of outer radii. Must match
                        the length of `inner_radii`.
    :type outer_radii: list of ints
    :param scan_dimensions: the dimensions of the scan, where the order is
                            (width, height). If set to (0, 0), an attempt
                            will be made to read the scan dimensions from
                            the data file.
    :type scan_dimensions: tuple of ints of length 2
    :param center: the center of the images, where the order is (x, y). If set
                   to (-1, -1), the center will be set to
                   (scan_dimensions[0] / 2, scan_dimensions[1] / 2).
    :type center: tuple of ints of length 2

    :return: The bins and the frequencies of the histogram.
    :rtype: a tuple of length 2 of lists
    """
    # create stem images
    imgs = _image.create_stem_images(reader.begin(), reader.end(),
                                     inner_radii, outer_radii,
                                     scan_dimensions, center)

    allBins = []
    allFreqs = []
    for inImage in imgs:
        bins = _image.get_container(inImage, numBins)
        freq = _image.create_stem_histogram(inImage, numBins, bins)
        bins = np.array(bins, copy=False)
        freq = np.array(freq, copy=False)
        allBins.append(bins)
        allFreqs.append(freq)

    return allBins, allFreqs

class ImageArray(np.ndarray):
    def __new__(cls, array, dtype=None, order=None):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)
        obj._image = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._image = getattr(obj, '_image', None)


def calculate_average(reader):
    """Create an average image of all the images.

    :param reader: the file reader that has already opened the data.
    :type reader: stempy.io.reader

    :return: The average image.
    :rtype: stempy.image.ImageArray
    """
    image =  _image.calculate_average(reader.begin(), reader.end())
    img = ImageArray(np.array(image, copy = False))
    img._image = image

    return img


def electron_count(reader, darkreference=None, number_of_samples=40,
                   background_threshold_n_sigma=4, xray_threshold_n_sigma=10,
                   threshold_num_blocks=1, scan_dimensions=(0, 0),
                   verbose=False, gain=None, apply_row_dark=False,
                   apply_row_dark_use_mean=True):
    """Generate a list of coordinates of electron hits.

    :param reader: the file reader that has already opened the data.
    :type reader: stempy.io.reader
    :param darkreference: the dark reference to subtract, potentially generated
                          via stempy.image.calculate_average().
    :type darkreference: stempy.image.ImageArray or stempy::Image<double>
    :param number_of_samples: the number of samples to take when calculating
                              the thresholds.
    :type number_of_samples: int
    :param background_threshold_n_sigma: N-Sigma used for calculating the
                                         background threshold.
    :type background_threshold_n_sigma: int
    :param xray_threshold_n_sigma: N-Sigma used for calculating the X-Ray
                                   threshold
    :type xray_threshold_n_sigma: int
    :param threshold_num_blocks: The number of blocks of data to use when
                                 calculating the threshold.
    :type threshold_num_blocks: int
    :param scan_dimensions: the dimensions of the scan, where the order is
                            (width, height). Required if `data` is a
                            numpy.ndarray.
    :type scan_dimensions: tuple of ints of length 2
    :param verbose: whether or not to print out verbose output.
    :type verbose: bool
    :param gain: the gain mask to apply. Must match the frame dimensions
    :type gain: numpy.ndarray (2D)
    :param apply_row_dark: whether to apply the row dark algorithm to the data.
    :type apply_row_dark: bool
    :param apply_row_dark_use_mean: whether to use mean (if True) or median
                                    (if False) in the row dark algorith. Only
                                    applicable if apply_row_dark is True.
    :param apply_row_dark_use_mean: bool

    :return: the coordinates of the electron hits for each frame.
    :rtype: SparseArray
    """
    if gain is not None:
        # Invert, as we will multiply in C++
        # It also must be a float32
        gain = np.power(gain, -1)
        gain = _safe_cast(gain, np.float32, 'gain')

    if isinstance(darkreference, np.ndarray):
        # Must be float32 for correct conversions
        darkreference = _safe_cast(darkreference, np.float32, 'dark reference')

    # Special case for threaded reader
    if isinstance(reader, (SectorThreadedReader, SectorThreadedMultiPassReader)):
        options = _image.ElectronCountOptions()

        options.dark_reference = darkreference
        options.threshold_number_of_blocks = threshold_num_blocks
        options.number_of_samples = number_of_samples
        options.background_threshold_n_sigma = background_threshold_n_sigma
        options.x_ray_threshold_n_sigma = xray_threshold_n_sigma
        options.gain = gain
        options.scan_dimensions = scan_dimensions
        options.verbose = verbose
        options.apply_row_dark_subtraction = apply_row_dark
        options.apply_row_dark_use_mean = apply_row_dark_use_mean

        data = _image.electron_count(reader, options)
    else:
        deprecation_message = (
            'Using a reader in electron_count() that is not a '
            'SectorThreadedReader or a SectorThreadedMultiPassReader is '
            'deprecated in stempy==1.1 and will be removed in stempy==1.2'
        )
        warnings.warn(deprecation_message, category=DeprecationWarning,
                      stacklevel=2)

        blocks = []
        for i in range(threshold_num_blocks):
            blocks.append(next(reader))

        if darkreference is not None and hasattr(darkreference, '_image'):
            darkreference = darkreference._image

        args = [[b._block for b in blocks]]
        if darkreference is not None:
            args.append(darkreference)

        args = args + [number_of_samples, background_threshold_n_sigma, xray_threshold_n_sigma]

        if gain is not None:
            args.append(gain)

        res = _image.calculate_thresholds(*args)

        background_threshold = res.background_threshold
        xray_threshold = res.xray_threshold

        if verbose:
            print('****Statistics for calculating electron thresholds****')
            print('number of samples:', res.number_of_samples)
            print('min sample:', res.min_sample)
            print('max sample:', res.max_sample)
            print('mean:', res.mean)
            print('variance:', res.variance)
            print('std dev:', res.std_dev)
            print('number of bins:', res.number_of_bins)
            print('x-ray threshold n sigma:', res.xray_threshold_n_sigma)
            print('background threshold n sigma:',
                res.background_threshold_n_sigma)
            print('optimized mean:', res.optimized_mean)
            print('optimized std dev:', res.optimized_std_dev)
            print('background threshold:', background_threshold)
            print('xray threshold:', xray_threshold)

        # Reset the reader
        reader.reset()

        options = _image.ElectronCountOptionsClassic()

        # Do we need to convert this to a numpy array?
        options.dark_reference = darkreference
        options.background_threshold = background_threshold
        options.x_ray_threshold = xray_threshold
        options.gain = gain
        options.scan_dimensions = scan_dimensions
        options.apply_row_dark_subtraction = apply_row_dark
        options.optimized_mean = res.optimized_mean
        options.apply_row_dark_use_mean = apply_row_dark_use_mean

        data = _image.electron_count(reader.begin(), reader.end(), options)

    # Convert to numpy array
    num_scans = len(data.data)
    frames_per_scan = len(data.data[0]) if data.data else 0

    np_data = np.empty((num_scans, frames_per_scan), dtype=object)
    for i, scan_frames in enumerate(data.data):
        for j, sparse_frame in enumerate(scan_frames):
            np_data[i, j] = np.array(sparse_frame, copy=False)

    metadata = _electron_counted_metadata_to_dict(data.metadata)

    kwargs = {
        'data': np_data,
        'scan_shape': data.scan_dimensions[::-1],
        'frame_shape': data.frame_dimensions,
        'metadata': {'electron_counting': metadata},
    }
    array = SparseArray(**kwargs)

    # Store a copy of the underlying C++ object in case we need it later
    array._electron_counted_data = data

    return array

def radial_sum(reader, center=(-1, -1), scan_dimensions=(0, 0)):
    """Generate a radial sum from which STEM images can be generated.

    :param reader: the file reader that has already opened the data.
    :type reader: stempy.io.reader
    :param center: the center of the image, where the order is (x, y). If set
                   to (-1, -1), the center will be set to
                   (scan_dimensions[0] / 2, scan_dimensions[1] / 2).
    :type center: tuple of ints of length 2
    :param scan_dimensions: the dimensions of the scan, where the order is
                            (width, height). If set to (0, 0), an attempt
                            will be made to read the scan dimensions from
                            the data file.
    :type scan_dimensions: tuple of ints of length 2

    :return: The generated radial sum.
    :rtype: numpy.ndarray
    """

    sum =  _image.radial_sum(reader.begin(), reader.end(), scan_dimensions,
                             center)

    return np.array(sum, copy=False)


def maximum_diffraction_pattern(reader, darkreference=None):
    """Generate an image of the maximum diffraction pattern.

    :param reader: the file reader that has already opened the data.
    :type reader: stempy.io.reader
    :param darkreference: the dark reference to subtract, potentially generated
                          via stempy.image.calculate_average().
    :type darkreference: stempy.image.ImageArray or stempy::Image<double>

    :return: the maximum diffraction pattern.
    :rtype: stempy.image.ImageArray
    """
    if darkreference is not None:
        darkreference = darkreference._image
        image = _image.maximum_diffraction_pattern(reader.begin(), reader.end(), darkreference)
    else:
        image = _image.maximum_diffraction_pattern(reader.begin(), reader.end())
    img = ImageArray(np.array(image, copy=False))
    img._image = image

    return img


def com_sparse(array, crop_to=None, init_center=None, replace_nans=True):
    """Compute center of mass (COM) for counted data directly from sparse (single)
        electron data. Empty frames will have COM set to NAN. There is an option to crop to a
        smaller region around the initial full frame COM to improve finding the center of the zero beam. If the
        cropped region has no counts in it then the frame is considered empty and COM will be NAN.
        
        :param array: A SparseArray of the electron counted data
        :type array: stempy.io.SparseArray
        :param crop_to: optional; The size of the region to crop around initial full frame COM for improved COM near
                        the zero beam
        :type crop_to: tuple of ints of length 2
        :param init_center: optional; The initial center to use before cropping. If this is not set then cropping will be applie around 
                            the center of mass of the each full frame.
        :type init_center: tuple of ints of length 2
        :param replace_nans: If true (default) empty frames will have their center of mass set as the mean of the center of mass
                             of the the entire data set. If this is False they will be set to np.NAN.
        :type replace_nans: bool
        :return: The center of mass in X and Y for each scan position. If a position
                has no data (len(electron_counts) == 0) then the center of the
                frame is used as the center of mass.
        :rtype: numpy.ndarray (2D)
        """
    com = np.zeros((2, array.num_scans), np.float32)
    for scan_position, frames in enumerate(array.data):
        # Combine the sparse arrays into one array.
        # This takes care of multiple frames per probe position
        ev = np.hstack(frames)

        if len(ev) > 0:
            x = ev // array.frame_shape[0]
            y = ev % array.frame_shape[1]
            mm0 = len(ev)
            
            if init_center is None:
                # Initialize center as full frame COM
                comx0 = np.sum(x) / mm0
                comy0 = np.sum(y) / mm0
            else:
                comx0 = init_center[0]
                comy0 = init_center[1]
            
            if crop_to is not None:
                # Crop around the initial center
                keep = (x > comx0 - crop_to[0]) & (x <= comx0 + crop_to[0]) & (y > comy0 - crop_to[1]) & (
                        y <= comy0 + crop_to[1])
                x = x[keep]
                y = y[keep]
                mm = len(x)
                if mm > 0:
                    # Some counts found inside cropped region
                    comx = np.sum(x) / mm
                    comy = np.sum(y) / mm
                else:
                    # No counts in cropped region.
                    # Set as NAN
                    comx = np.nan
                    comy = np.nan
            else:
                # Center of mass of the full frame
                comx = np.sum(x) / mm0
                comy = np.sum(y) / mm0
            
            com[:, scan_position] = (comy, comx)  # save the comx and comy. Needs to be reversed (comy, comx)
        else:
            com[:, scan_position] = (np.nan, np.nan)  # empty frame

    com = com.reshape((2, *array.scan_shape))
    
    return com


def filter_bad_sectors(com, cut_off):
    """ Missing sectors of data can greatly affect the center of mass calculation. This
    function attempts to fix that issue. Usually, the problem will be evident by a
    bimodal histogram for the horizontal center of mass. Set the cut_off to a value between
    the real values and the outliers. cut_off is a tuple where values below cut_off[0]
    or above cut_off[1] will be set to the local median.

    :param com: The center of mass for the vertical and horizontal axes
    :type com: np.ndarray (3d)
    :param cut_off: The lower and upper cutoffs as a 2-tuple
    :type cut_off: 2-tuple

    :rtype numpy.ndarray (3d)
    """
    _ = (com[1, :, :] > cut_off[1]) | (com[1, :, :] < cut_off[0])
    com[0, _] = np.nan
    com[1, _] = np.nan

    x, y = np.where(_)

    for ii, jj in zip(x, y):
        com[0, ii, jj] = np.nanmedian(com[0, ii - 1:ii + 2, jj - 1:jj + 2])
        com[1, ii, jj] = np.nanmedian(com[1, ii - 1:ii + 2, jj - 1:jj + 2])
    return com

def com_dense(frames):
    """Compute the center of mass for a set of dense 2D frames.

    :param frames: The frames to calculate the center of mass from
    :type frames: numpy.ndarray (2D or 3D)

    :return: The center of mass along each axis of set of frames
    :rtype: np.ndarray

    """
    # Make 3D if its only 2D
    if frames.ndim == 2:
        frames = frames[None, :, :]

    YY, XX = np.mgrid[0:frames.shape[1], 0:frames.shape[2]]
    com = np.zeros((2, frames.shape[0]), dtype=np.float64)
    for ii, dp in enumerate(frames):
        mm = dp.sum()
        com_x = np.sum(XX * dp)
        com_y = np.sum(YY * dp)
        com[:, ii] = (com_x / mm, com_y / mm)
    return com


def calculate_sum_sparse(electron_counts, frame_dimensions):
    """Compute a diffraction pattern from sparse electron counted data.

    :param electron_counts: A vector of electron positions flattened. Each
                            pixel can only be a 1 (electron) or a 0
                            (no electron).
    :type electron_counts: numpy.ndarray (1D)
    :param frame_dimensions: The shape of the detector.
    :type frame_dimensions: tuple of ints of length 2

    :return: A summed diffraction pattern.
    :rtype: numpy.ndarray
    """
    dp = np.zeros((frame_dimensions[0] * frame_dimensions[1]), '<u8')
    for ev in electron_counts:
        for ee in ev:
            dp[ee] += 1
    dp = dp.reshape(frame_dimensions)
    return dp


def radial_sum_sparse(sa, center):
    """Radial sum of sparse electron counted data

    :param sa: A SparseArray data set.
    :type sa: stempy.io.SparseArray
    :param center: The center of the diffraction pattern in pixels.
    :type center: tuple of ints of length 2

    :return: A ndarray of the radial sum of shape (scan_shape[0],
             scan_shape[0], max(frame_dimensions)/2
    :rtype: numpy.ndarray
    """

    num_bins = int(max(sa.frame_shape) / 2)
    r_sum = np.zeros((sa.num_scans, num_bins), dtype='<i8')

    for ii, frames in enumerate(sa.data):
        # Loop over all scan positions
        for jj, events in enumerate(frames):
            # Loop over all frames in this scan position
            x, y = np.unravel_index(events, sa.frame_shape)
            r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            hh, hhx = np.histogram(r, bins=range(0, num_bins + 1))
            r_sum[ii, :] += hh
    r_sum = r_sum.reshape((sa.scan_shape[0], sa.scan_shape[1], num_bins))

    return r_sum


def _safe_cast(array, dtype, name=None):
    # Cast the array to a different dtype, ensuring no loss of
    # precision. Otherwise, an exception will be raised.
    if name is None:
        name = 'array'

    new_array = array.astype(dtype)
    if np.any(new_array.astype(array.dtype) != array):
        msg = f'Cannot cast {name} to {dtype} without loss of precision'
        raise ValueError(msg)
    return new_array


def phase_from_com(com, theta=0, flip=False, reg=1e-10):
    """Integrate 4D-STEM centre of mass (DPC) measurements to calculate
    object phase. Assumes a three dimensional array com, with the final
    two dimensions corresponding to the image and the first dimension
    of the array corresponding to the y and x centre of mass respectively.
    Note this version of the reconstruction is not quantitative.

    Thanks to the py4DSTEM project and author Hamish Brown.

    :param com: The center of mass for each frame as a 3D array of size [2, M, N]
    :type com: numpy.ndarray, 3D
    :param theta: The angle between real space and reciprocal space in radians
    type theta: float
    :param flip: Whether to flip the com direction to account for a mirror across the vertical axis.
    type flip: bool
    :param reg: A regularization parameter
    type reg: float

    :return: A 2D ndarray of the DPC phase.
    :rtype: numpy.ndarray

    """
    # Perform rotation and flipping if needed (from py4dstem)
    CoMx = com[0, :, :]
    CoMy = com[1, :, :]
    if flip:
        CoMx_rot = CoMx * np.cos(theta) + CoMy * np.sin(theta)
        CoMy_rot = CoMx * np.sin(theta) - CoMy * np.cos(theta)
    else:
        CoMx_rot = CoMx * np.cos(theta) - CoMy * np.sin(theta)
        CoMy_rot = CoMx * np.sin(theta) + CoMy * np.cos(theta)

    # Get shape of arrays
    ny, nx = com.shape[1:]

    # Calculate Fourier coordinates for array
    ky, kx = [np.fft.fftfreq(x) for x in [ny, nx]]

    # Calculate numerator and denominator expressions for solution of
    # phase from centre of mass measurements
    numerator = ky[:, None] * np.fft.fft2(CoMx_rot) + kx[None, :] * np.fft.fft2(CoMy_rot)
    denominator = 2 * np.pi * 1j * ((kx ** 2)[None, :] + (ky ** 2)[:, None]) + reg
    # Avoid a divide by zero for the origin of the Fourier coordinates
    numerator[0, 0] = 0
    denominator[0, 0] = 1

    # Return real part of the inverse Fourier transform
    return np.real(np.fft.ifft2(numerator / denominator))


def _electron_counted_metadata_to_dict(metadata):
    # Convert the electron counted metadata to a python dict
    attributes = [
        'threshold_calculated',
        'background_threshold',
        'x_ray_threshold',
        'number_of_samples',
        'min_sample',
        'max_sample',
        'mean',
        'variance',
        'std_dev',
        'number_of_bins',
        'x_ray_threshold_n_sigma',
        'background_threshold_n_sigma',
        'optimized_mean',
        'optimized_std_dev',
    ]

    ret = {}
    for name in attributes:
        ret[name] = getattr(metadata, name)

    return ret

def virtual_darkfield(array, centers_x, centers_y, radii, plot=False):
    """Calculate a virtual dark field image from a set of round virtual apertures in diffraction space.
    Each aperture is defined by a center and radius and the final image is the sum of all of them.
    
    :param array: The SparseArray
    :type array: SparseArray
    
    :param centers_x: The center of each round aperture as the row locations
    :type centers_x: number or iterable
    
    :param centers_y: The center of each round aperture as the column locations
    :type centers_y: number or iterable
    
    :param radii: The radius of each aperture.
    :type radii: number or iterable
    
    :param plot: If set to True then the apertures are plotted as circles using plot_virtual_darkfield
    :rype plot: bool
    
    :rtype: np.ndarray
    
    :example:
    >>> sp = stempy.io.load_electron_counts('file.h5')
    >>> dp2 = stempy.image.virtual_darkfield(sp, (288, 260), (288, 160), (10, 10)) # 2 apertures
    >>> dp1 = stempy.image.virtual_darkfield(sp, 260, 160, 10) # 1 aperture
    
    """
    
    # Change to iterable if single value
    if isinstance(centers_x, (int, float)):
         centers_x = (centers_x,)
    if isinstance(centers_y, (int, float)):
         centers_y = (centers_y,)
    if isinstance(radii, (int, float)):
         radii = (radii,)

    rs_image = np.zeros((array.shape[0] * array.shape[1],), dtype=array.dtype)
    for ii, events in enumerate(array.data):
        for ev in events:
            ev_rows = ev // array.frame_shape[0]
            ev_cols = ev % array.frame_shape[1]
            for cc_0, cc_1, rr in zip(centers_x, centers_y, radii):
                dist = np.sqrt((ev_rows - cc_1)**2 + (ev_cols - cc_0)**2)
                rs_image[ii] += len(np.where(dist < rr)[0])
    rs_image = rs_image.reshape(array.shape[0:2])
    
    return rs_image

def plot_virtual_darkfield(image, centers_x, centers_y, radii, axes=None):
    """Plot circles on the diffraction pattern corresponding to the position and size of virtual dark field apertures.
    This has the same input as stempy.image.virtual_darkfield so users can check their input is physically correct.
    
    :param image: The diffraciton pattern to plot over
    :type image: np.ndarray, 2D
    
    :param centers_x: The center of each round aperture as the row locations
    :type centers_x: iterable
    
    :param centers_y: The center of each round aperture as the column locations
    :type centers_y: iterable
    
    :param radii: The radius of each aperture. This has to be in the form (r0, )
    :type radii: iterable
    
    :param axes: A matplotlib axes instance to use for the plotting. If None then a new plot is created.
    :type axes: matplotlib.axes._subplots.AxesSubplot
    
    :rtype: matplotlib.axes._subplots.AxesSubplot
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle

    # Change to iterable if single value
    if isinstance(centers_x, (int, float)):
         centers_x = (centers_x,)
    if isinstance(centers_y, (int, float)):
         centers_y = (centers_y,)
    if isinstance(radii, (int, float)):
         radii = (radii,)
    
    if not axes:
        fg, axes = plt.subplots(1, 1)
    
    axes.imshow(image, cmap='magma', norm=LogNorm())
    
    # Place a circle at each apertue location
    for cc_0, cc_1, rr in zip(centers_x, centers_y, radii):
        C = Circle((cc_0, cc_1), rr, fc='none', ec='c')
        axes.add_patch(C)
    
    return axes

def mask_real_space(array, mask):
    """Calculate a diffraction pattern from an arbitrary set of positions defined in a mask in real space
    
    :param array: The sparse dataset
    :type array: SparseArray
    
    :param mask: The mask to apply with 0 for probe positions to ignore and 1 for probe positions to include in the sum. Must have the same scan shape as array
    :type mask: np.ndarray
    
    :rtype: np.ndarray
    """
    assert array.scan_shape[0] == mask.shape[0] and array.scan_shape[1] == mask.shape[1]
    dp_mask = np.zeros(np.prod(array.frame_shape), array.dtype)
    mr = np.ravel_multi_index(np.where(mask), array.scan_shape)
    for events in array.data[mr, :]:
        for ev in events:
            dp_mask[ev] += 1
    dp_mask = dp_mask.reshape(array.frame_shape)
    return dp_mask
