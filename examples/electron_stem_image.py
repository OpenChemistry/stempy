import glob
import h5py
from stempy import io, image
import numpy as np
from PIL import Image

def save_img(stem_image_data, name, scan_dimensions):
    min = np.min(stem_image_data)
    max = np.max(stem_image_data)

    stem_image_data = stem_image_data.reshape(scan_dimensions)
    stem_image_data = np.interp(stem_image_data, [min, max], [0, 256])
    stem_image_data = stem_image_data.astype(np.uint8)
    img = Image.fromarray(stem_image_data)
    img.save(name)

with h5py.File('stem_image.h5', 'r') as rf:
    frames = rf['/electron_events/frames'][()]
    attrs = rf['/electron_events/frames'].attrs
    frame_dimensions = (attrs['Nx'], attrs['Ny'])

    attrs = rf['/electron_events/scan_positions'].attrs
    scan_dimensions = (attrs['Nx'], attrs['Ny'])

num_pixels = frame_dimensions[0] * frame_dimensions[1]

inner_radius = 40
outer_radius = 288

img = image.create_stem_images(frames, inner_radius, outer_radius,
                               scan_dimensions,
                               frame_dimensions=frame_dimensions)[0]

save_img(img, 'img.png', scan_dimensions)
