import h5py
import sys
from stempy import io, image
import numpy as np
from PIL import Image

def save_img(stem_image_data, name):
    min = np.min(stem_image_data)
    max = np.max(stem_image_data)

    stem_image_data = stem_image_data.reshape((160, 160))
    stem_image_data = np.interp(stem_image_data, [min, max], [0, 256])
    stem_image_data = stem_image_data.astype(np.uint8)
    img = Image.fromarray(stem_image_data)
    img.save(name)

if(len(sys.argv)!=2):
    print ("python3 reader_hdf5.py <file name>")
    exit(0)

filename=sys.argv[1]
h5file= h5py.File(filename, 'r')

inner_radii = [0, 40]
outer_radii = [288, 288]

imgs = image.create_stem_images(h5file, inner_radii, outer_radii,
                                scan_dimensions=(160, 160))

for i, img in enumerate(imgs):
    suffix = str(inner_radii[i]) + '_' + str(outer_radii[i]) + '.png'
    save_img(img, 'img_' + suffix)
