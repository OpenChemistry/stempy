from stempy import _image
import numpy as np
from collections import namedtuple

def create_stem_image(blocks, width, height,  inner_radius, outer_radius):
    img =  _image.create_stem_image([b._block for b in blocks],
                                    width, height,  inner_radius, outer_radius)

    image = namedtuple('STEMImage', ['bright', 'dark'])
    image.bright = np.array(img.bright, copy = False)
    image.dark = np.array(img.dark, copy = False)

    return image
