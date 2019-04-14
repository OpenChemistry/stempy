from stempy import _image
import numpy as np
from collections import namedtuple

def create_stem_image(blockiterator, width, height,  inner_radius,
                      outer_radius):
    img =  _image.create_stem_image(blockiterator,
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

def calculate_average(blocks):
    image =  _image.calculate_average([b._block for b in blocks])
    img = ImageArray(np.array(image, copy = False))
    img._image = image

    return img

def electron_count(blocks, rows, columns, darkreference,  number_of_samples=40,
                   background_threshold_n_sigma=4, xray_threshold_n_sigma=10):
    events = _image.electron_count([b._block for b in blocks], rows, columns,
                                   darkreference._image, number_of_samples,
                                   background_threshold_n_sigma, xray_threshold_n_sigma)

    # Convert to numpy and return
    return np.array([np.array(x) for x in events])
