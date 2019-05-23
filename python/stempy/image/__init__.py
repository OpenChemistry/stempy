from stempy import _image
import numpy as np

def create_stem_images(reader, inner_radii,
                       outer_radii, width=0, height=0,
                       center_x=-1, center_y=-1):
    imgs = _image.create_stem_images(reader.begin(), reader.end(),
                                     inner_radii, outer_radii, width, height,
                                     center_x, center_y)

    images = []
    for img in imgs:
        images.append(np.array(img, copy=False))

    return np.array(images, copy=False)

# This one exists for backward compatibility
def create_stem_image(reader, inner_radius, outer_radius, width=0, height=0,
                      center_x=-1, center_y=-1):
    return create_stem_images(reader, (inner_radius,), (outer_radius,),
                              width, height, center_x, center_y)[0]

def create_stem_images_sparse(data, inner_radius, outer_radius,
                              width, height, frame_width, frame_height,
                              center_x=-1, center_y=-1):
    imgs = _image.create_stem_images_sparse(data, inner_radius, outer_radius,
                                            width, height, frame_width,
                                            frame_height, center_x, center_y)

    images = []
    for img in imgs:
        images.append(np.array(img, copy=False))

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
