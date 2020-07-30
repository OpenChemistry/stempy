from stempy import _image
from stempy import _io
from stempy.io import get_hdf5_reader, ReaderMixin, PyReader, SectorThreadedReader

from collections import namedtuple
import deprecation
import h5py
import numpy as np

def create_stem_images(input, inner_radii, outer_radii, scan_dimensions=(0, 0),
                       center=(-1, -1), frame_dimensions=None, frame_offset=0):
    """Create a series of stem images from the input.

    :param input: the file reader that has already opened the data, or an
                  open h5py file, or an ElectronCountedData namedtuple
                  containing sparse data, or a numpy.ndarray of either the
                  sparse or the raw data (if the frame_dimensions argument
                  is supplied, numpy.ndarray is inferred to be sparse data).
    :type input: stempy.io.reader, an h5py file, ElectronCountedData, or
                 numpy.ndarray
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
        imgs = _image.create_stem_images(input._electron_counted_data,
                                         inner_radii, outer_radii, center)
    elif all([hasattr(input, x) for x in ecd_attrs]):
        # Handle electron counted data without C++ object
        imgs = _image.create_stem_images(input.data, inner_radii, outer_radii,
                                         input.scan_dimensions,
                                         input.frame_dimensions, center, 0)
    elif isinstance(input, np.ndarray):
        # The presence of frame dimensions implies it is sparse data
        if frame_dimensions is not None:
            # Handle sparse data
            imgs = _image.create_stem_images(input, inner_radii, outer_radii,
                                             scan_dimensions, frame_dimensions,
                                             center, frame_offset)
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

@deprecation.deprecated(
    deprecated_in='1.0', removed_in='1.1',
    details='Use stempy.image.create_stem_images() instead.')
def create_stem_image(reader, inner_radius, outer_radius,
                      scan_dimensions=(0, 0), center=(-1, -1)):
    """Create a stem image from the input.

    :param reader: the file reader that has already opened the data.
    :type reader: stempy.io.reader or an h5py file
    :param inner_radius: the inner radius to use.
    :type inner_radius: int
    :param outer_radius: the outer radius to use.
    :type outer_radius: int
    :param scan_dimensions: the dimensions of the scan, where the order is
                            (width, height). If set to (0, 0), an attempt
                            will be made to read the scan dimensions from
                            the data file.
    :type scan_dimensions: tuple of ints of length 2
    :param center: the center of the image, where the order is (x, y). If set
                   to (-1, -1), the center will be set to
                   (scan_dimensions[0] / 2, scan_dimensions[1] / 2).
    :type center: tuple of ints of length 2

    :return: The STEM image that was generated.
    :rtype: numpy.ndarray
    """
    return create_stem_images(reader, inner_radius, outer_radius,
                              scan_dimensions, center)[0]

@deprecation.deprecated(
    deprecated_in='1.0', removed_in='1.1',
    details='Use stempy.image.create_stem_images() instead.')
def create_stem_images_sparse(data, inner_radii, outer_radii,
                              scan_dimensions=None, frame_dimensions=None,
                              center=(-1, -1), frame_offset=0):
    """Create a series of stem images from sparsified data.

    :param data: the input sparsified data.
    :type data: ElectronCountedData or numpy.ndarray
    :param inner_radii: a list of inner radii. Must match
                        the length of `outer_radii`.
    :type inner_radii: list of ints
    :param outer_radii: a list of outer radii. Must match
                        the length of `inner_radii`.
    :type outer_radii: list of ints
    :param scan_dimensions: the dimensions of the scan, where the order is
                            (width, height). Required if `data` is a
                            numpy.ndarray.
    :type scan_dimensions: tuple of ints of length 2
    :param frame_dimensions: the dimensions of each frame, where the order is
                             (width, height). Required if `data` is a
                             numpy.ndarray.
    :type frame_dimensions: tuple of ints of length 2
    :param center: the center of the images, where the order is (x, y). If set
                   to (-1, -1), the center will be set to
                   (scan_dimensions[0] / 2, scan_dimensions[1] / 2).
    :type center: tuple of ints of length 2
    :param frame_offset: the amount by which to offset the frame.
    :type frame_offset: int

    :return: A numpy array of the STEM images.
    :rtype: numpy.ndarray
    """
    return create_stem_images(data, inner_radii, outer_radii,
                              scan_dimensions, center, frame_dimensions,
                              frame_offset)

@deprecation.deprecated(
    deprecated_in='1.0', removed_in='1.1',
    details='Use stempy.image.create_stem_images() instead.')
def create_stem_image_sparse(data, inner_radius, outer_radius,
                             scan_dimensions=None, frame_dimensions=None,
                             center=(-1, -1), frame_offset=0):
    """Create a stem image from sparsified data.

    :param data: the input sparsified data.
    :type data: ElectronCountedData or numpy.ndarray
    :param inner_radius: the inner radius to use.
    :type inner_radius: int
    :param outer_radius: the outer radius to use.
    :type outer_radius: int
    :param scan_dimensions: the dimensions of the scan, where the order is
                            (width, height). Required if `data` is a
                            numpy.ndarray.
    :type scan_dimensions: tuple of ints of length 2
    :param frame_dimensions: the dimensions of each frame, where the order is
                             (width, height). Required if `data` is a
                             numpy.ndarray.
    :type frame_dimensions: tuple of ints of length 2
    :param center: the center of the images, where the order is (x, y). If set
                   to (-1, -1), the center will be set to
                   (scan_dimensions[0] / 2, scan_dimensions[1] / 2).
    :type center: tuple of ints of length 2
    :param frame_offset: the amount by which to offset the frame.
    :type frame_offset: int

    :return: The STEM image that was generated.
    :rtype: numpy.ndarray
    """
    return create_stem_images(data, inner_radius, outer_radius,
                              scan_dimensions, center, frame_dimensions,
                              frame_offset)[0]

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


def electron_count(reader, darkreference, number_of_samples=40,
                   background_threshold_n_sigma=4, xray_threshold_n_sigma=10,
                   threshold_num_blocks=1, scan_dimensions=(0, 0),
                   verbose=False, gain=None):
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

    :return: the coordinates of the electron hits for each frame.
    :rtype: ElectronCountedData (named tuple with fields 'data',
            'scan_dimensions', and 'frame_dimensions')
    """

    # Special case for threaded reader
    if isinstance(reader, SectorThreadedReader):
        args = [reader, darkreference]

        # add the gain arg if we have been given one.
        if gain is not None:
            # Invert as we will multiple in C++
            gain = np.power(gain, -1)
            args.append(gain)

        # Now add the other args
        args = args + [threshold_num_blocks, number_of_samples,
                       background_threshold_n_sigma, xray_threshold_n_sigma,
                       scan_dimensions, verbose]

        data = _image.electron_count(*args)
    else:
        blocks = []
        for i in range(threshold_num_blocks):
            blocks.append(next(reader))

        if hasattr(darkreference, '_image'):
            darkreference = darkreference._image

        args = [[b._block for b in blocks], darkreference]

        if gain is not None:
            args.append(gain)

        args = args + [number_of_samples, background_threshold_n_sigma, xray_threshold_n_sigma]


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

        args = [reader.begin(), reader.end(), darkreference]
        if gain is not None:
            args.append(gain)


        args = args + [background_threshold, xray_threshold, scan_dimensions]

        data = _image.electron_count(*args)

    electron_counted_data = namedtuple('ElectronCountedData',
                                       ['data', 'scan_dimensions',
                                        'frame_dimensions'])

    # Convert to numpy array
    electron_counted_data.data = np.array([np.array(x, copy=False) for x in data.data], dtype=np.object)
    electron_counted_data.scan_dimensions = data.scan_dimensions
    electron_counted_data.frame_dimensions = data.frame_dimensions

    # Store a copy of the underlying C++ object in case we need it later
    electron_counted_data._electron_counted_data = data

    return electron_counted_data

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


def com_sparse(electron_counts, frame_dimensions):
    """Compute center of mass for counted data directly from sparse (single)
        electron data.

        :param electron_counts: A vector of electron positions flattened. Each
                                pixel can only be a 1 (electron) or a 0
                                (no electron).
        :type electron_counts: numpy.ndarray (1D)
        :param frame_dimensions: The shape of the detector.
        :type frame_dimensions: tuple of ints of length 2

        :return: The center of mass in X and Y for each scan position. If a position
                has no data (len(electron_counts) == 0) then the center of the
                frame is used as the center of mass.
        :rtype: numpy.ndarray (2D)
        """
    frame_center = [int(ii/2)-1 for ii in frame_dimensions]
    com = np.zeros((2, len(electron_counts)), 'f')
    for ii, ev in enumerate(electron_counts):
        if len(ev) > 0:
            x, y = np.unravel_index(ev, frame_dimensions)
            mm = len(ev)
            com_x = np.sum(x) / mm
            com_y = np.sum(y) / mm
            com[:, ii] = (com_y, com_x)
        else:
            com[:, ii] = frame_center
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
    dp = np.zeros(frame_dimensions, '<u8')
    for ii, ev in enumerate(electron_counts):
        x, y = np.unravel_index(ev, dp.shape)
        dp[x, y] += 1

    return dp


def radial_sum_sparse(electron_counts, scan_dimensions, frame_dimensions,
                      center):
    """Radial sum of sparse (single) electron counted data

    :param electron_counts: A vector of electron positions flattened. Each
                            pixel can only be a 1 (electron) or a 0
                            (no electron).
    :type electron_counts: numpy.ndarray (1D)
    :param scan_dimensions: The number of X and Y scan positions in pixels.
    :type scan_dimensions: tuple of ints of length 2
    :param frame_dimensions: The shape of the detector.
    :type frame_dimensions: tuple of ints of length 2
    :param center: The center of the diffraction pattern in pixels.
    :type center: tuple of ints of length 2

    :return: A ndarray of the radial sum of shape (scan_dimensions[0],
             scan_dimensions[0], max(frame_dimensions/2)
    :rtype: numpy.ndarray
    """
    num_bins = int(max(frame_dimensions) / 2)
    r_sum = np.zeros((scan_dimensions[0] * scan_dimensions[1], num_bins),
                     dtype='<u8')

    for ii, ev in enumerate(electron_counts):
        x, y = np.unravel_index(ev, frame_dimensions)
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        hh, hhx = np.histogram(r, bins=range(0, num_bins + 1))
        r_sum[ii, :] = hh
    r_sum = r_sum.reshape((scan_dimensions[0], scan_dimensions[1], num_bins))

    return r_sum
