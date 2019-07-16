from stempy import _image
from stempy import _io
import numpy as np
import h5py


def get_hdf5_reader(h5file):
    # the initialization is at the io.cpp
    dset_frame=h5file['frames']
    dset_frame_shape=dset_frame.shape
    totalImgNum=dset_frame_shape[0]
    imagewidth=dset_frame_shape[1]
    imageheight=dset_frame_shape[2]
    
    dset_stem_shape=h5file['stem/images'].shape
    scanwidth=dset_stem_shape[1]
    scanheight=dset_stem_shape[2]
    
    # the number here are same with the smallscan data
    blocksize=32
    blockNumInFile=25
    image_numbers = []
    
    h5reader = _io._h5reader(dset_frame, image_numbers, imagewidth, imageheight, scanwidth, scanheight, blocksize, blockNumInFile, totalImgNum)
    return h5reader

def create_stem_images(input, inner_radii,
                       outer_radii, width=0, height=0,
                       center_x=-1, center_y=-1):
    # check if the input is the hdf5 dataset
    if(isinstance(input, h5py._hl.files.File)):
        reader = get_hdf5_reader(input)
        print("hdf5 reader is used")
    else:
        reader = input
        print("file stream reader is used")
    
    imgs = _image.create_stem_images(reader.begin(), reader.end(),
                                     inner_radii, outer_radii, width, height,
                                     center_x, center_y)

    images = [np.array(img, copy=False) for img in imgs]
    return np.array(images, copy=False)

def create_stem_histogram(numBins, reader, inner_radii,
                          outer_radii, width=0, height=0,
                          center_x=-1, center_y=-1):
    # create stem images
    imgs = _image.create_stem_images(reader.begin(), reader.end(),
                                     inner_radii, outer_radii, width, height,
                                     center_x, center_y)

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
def create_stem_image(reader, inner_radius, outer_radius, width=0, height=0,
                      center_x=-1, center_y=-1):
    return create_stem_images(reader, (inner_radius,), (outer_radius,),
                              width, height, center_x, center_y)[0]

def create_stem_images_sparse(data, inner_radii, outer_radii,
                              width, height, frame_width, frame_height,
                              center_x=-1, center_y=-1):
    imgs = _image.create_stem_images_sparse(data, inner_radii, outer_radii,
                                            width, height, frame_width,
                                            frame_height, center_x, center_y)

    images = [np.array(img, copy=False) for img in imgs]
    return np.array(images, copy=False)

def create_stem_image_sparse(data, inner_radius, outer_radius,
                             width, height, frame_width, frame_height,
                             center_x=-1, center_y=-1):
    return create_stem_images_sparse(data, [inner_radius], [outer_radius],
                                     width, height, frame_width, frame_height,
                                     center_x, center_y)[0]

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
                   threshold_num_blocks=1, scan_width=0, scan_height=0):

    blocks = []
    for i in range(threshold_num_blocks):
        blocks.append(next(reader))

    background_threshold, xray_threshold = _image.calculate_thresholds(
        [b._block for b in blocks], darkreference._image, number_of_samples,
        background_threshold_n_sigma, xray_threshold_n_sigma)

    # Reset the reader
    reader.reset()

    events = _image.electron_count(reader.begin(), reader.end(), darkreference._image,
                                   background_threshold, xray_threshold, scan_width, scan_height)

    # Convert to numpy and return
    return np.array([np.array(x) for x in events])

def radial_sum(reader, center_x=-1, center_y=-1, scan_width=0, scan_height=0):
    sum =  _image.radial_sum(reader.begin(), reader.end(), scan_width, scan_height,
                              center_x, center_y)

    return np.array(sum, copy=False)
