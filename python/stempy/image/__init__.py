from stempy import _image
import numpy as np

def create_stem_images(reader, inner_radii,
                       outer_radii, width=0, height=0,
                       center_x=-1, center_y=-1):
    imgs = _image.create_stem_images(reader.begin(), reader.end(),
                                     inner_radii, outer_radii, width, height,
                                     center_x, center_y)

    images = [np.array(img, copy=False) for img in imgs]
    return np.array(images, copy=False)

def create_stem_histogram(num_bins, num_hist, reader, inner_radii,
                          outer_radii, width=0, height=0,
                          center_x=-1, center_y=-1):
    # create stem images
    imgs = _image.create_stem_images(reader.begin(), reader.end(),
                                     inner_radii, outer_radii, width, height,
                                     center_x, center_y)

    all_bins = []
    all_freqs = []

    for in_image in imgs:
        bins = _image.get_container(in_image, num_bins)
        # each image can have numHist histograms, but all with the same bins
        freq = _image.create_stem_histogram(in_image, num_hist, num_bins, bins)
        bins = np.array(bins, copy=False)
        freq = np.array(freq, copy=False)
        all_bins.append(bins)
        all_freqs.append(freq)

    return all_bins, all_freqs

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
