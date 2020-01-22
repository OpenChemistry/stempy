from stempy import _image
from stempy import _io
import numpy as np
import h5py
from collections import namedtuple
from stempy.io import get_hdf5_reader

def create_stem_images(input, inner_radii, outer_radii, scan_dimensions=(0, 0),
                       center=(-1, -1)):
    # check if the input is the hdf5 dataset
    if(isinstance(input, h5py._hl.files.File)):
        reader = get_hdf5_reader(input)
    else:
        reader = input

    imgs = _image.create_stem_images(reader.begin(), reader.end(),
                                     inner_radii, outer_radii,
                                     scan_dimensions, center)

    images = [np.array(img, copy=False) for img in imgs]
    return np.array(images, copy=False)

def create_stem_histogram(numBins, reader, inner_radii,
                          outer_radii, scan_dimensions=(0, 0),
                          center=(-1, -1)):
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

# This one exists for backward compatibility
def create_stem_image(reader, inner_radius, outer_radius,
                      scan_dimensions=(0, 0), center=(-1, -1)):
    return create_stem_images(reader, (inner_radius,), (outer_radius,),
                              scan_dimensions, center)[0]

def create_stem_images_sparse(data, inner_radii, outer_radii,
                              scan_dimensions=(0, 0), frame_dimensions=(0, 0),
                              center=(-1, -1), frame_offset=0):
    """
    scan_dimensions and frame_dimensions are required if
    "data" is of type np.ndarray.
    """
    if not isinstance(data, np.ndarray):
        # Assume it is an ElectronCountedData named tuple
        imgs = _image.create_stem_images_sparse(data._electron_counted_data,
                                                inner_radii, outer_radii,
                                                center)
    else:
        imgs = _image.create_stem_images_sparse(data, inner_radii, outer_radii,
                                                scan_dimensions,
                                                frame_dimensions, center,
                                                frame_offset)

    images = [np.array(img, copy=False) for img in imgs]
    return np.array(images, copy=False)

def create_stem_image_sparse(data, inner_radius, outer_radius,
                             scan_dimensions=(0, 0), frame_dimensions=(0, 0),
                             center=(-1, -1), frame_offset=0):
    return create_stem_images_sparse(data, [inner_radius], [outer_radius],
                                     scan_dimensions, frame_dimensions, center,
                                     frame_offset)[0]

class ImageArray(np.ndarray):
    def __new__(cls, array, dtype=None, order=None):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)
        obj._image  = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._image = getattr(obj, '_image', None)

def calculate_average(reader):
    image =  _image.calculate_average(reader.begin(), reader.end())
    img = ImageArray(np.array(image, copy = False))
    img._image = image

    return img

def electron_count(reader, darkreference, number_of_samples=40,
                   background_threshold_n_sigma=4, xray_threshold_n_sigma=10,
                   threshold_num_blocks=1, scan_dimensions=(0, 0),
                   verbose=False):

    blocks = []
    for i in range(threshold_num_blocks):
        blocks.append(next(reader))

    if hasattr(darkreference, '_image'):
        darkreference = darkreference._image

    res = _image.calculate_thresholds(
        [b._block for b in blocks], darkreference, number_of_samples,
        background_threshold_n_sigma, xray_threshold_n_sigma)

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

    data = _image.electron_count(reader.begin(), reader.end(),
                                 darkreference, background_threshold,
                                 xray_threshold, scan_dimensions)

    electron_counted_data = namedtuple('ElectronCountedData',
                                       ['data', 'scan_dimensions',
                                        'frame_dimensions'])

    # Convert to numpy array
    electron_counted_data.data = np.array([np.array(x) for x in data.data])
    electron_counted_data.scan_dimensions = data.scan_dimensions
    electron_counted_data.frame_dimensions = data.frame_dimensions

    # Store a copy of the underlying C++ object in case we need it later
    electron_counted_data._electron_counted_data = data

    return electron_counted_data

def radial_sum(reader, center=(-1, -1), scan_dimensions=(0, 0)):
    sum =  _image.radial_sum(reader.begin(), reader.end(), scan_dimensions,
                             center)

    return np.array(sum, copy=False)


def maximum_diffraction_pattern(reader, darkreference=None):
    if darkreference is not None:
        darkreference = darkreference._image
        image = _image.maximum_diffraction_pattern(reader.begin(), reader.end(), darkreference)
    else:
        image = _image.maximum_diffraction_pattern(reader.begin(), reader.end())
    img = ImageArray(np.array(image, copy = False))
    img._image = image

    return img
