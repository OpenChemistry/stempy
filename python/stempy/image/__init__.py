from stempy import _image
import numpy as np
from collections import namedtuple

def create_stem_image(reader, width, height,  inner_radius,
                      outer_radius):
    img =  _image.create_stem_image(reader.begin(), reader.end(),
                                    width, height,  inner_radius, outer_radius)

    image = namedtuple('STEMImage', ['bright', 'dark'])
    image.bright = np.array(img.bright, copy = False)
    image.dark = np.array(img.dark, copy = False)

    return image

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

def electron_count(reader, rows, columns, darkreference,  number_of_samples=40,
                   background_threshold_n_sigma=4, xray_threshold_n_sigma=10,
                   threshold_num_blocks=1):

    blocks = []
    for i in range(threshold_num_blocks):
        blocks.append(next(reader))

    background_threshold, xray_threshold = _image.calculate_thresholds(
        [b._block for b in blocks], darkreference._image, number_of_samples,
        background_threshold_n_sigma, xray_threshold_n_sigma)

    # Reset the reader
    reader.reset()

    events = _image.electron_count(reader.begin(), reader.end(), rows, columns,
                                   darkreference._image, background_threshold,
                                   xray_threshold)

    # Convert to numpy and return
    return np.array([np.array(x) for x in events])
