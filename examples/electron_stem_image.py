import glob
import h5py
from stempy import io, image
import numpy as np
from PIL import Image

def save_img(stem_image_data, name, width, height):
    min = np.min(stem_image_data)
    max = np.max(stem_image_data)

    stem_image_data = stem_image_data.reshape((width, height))
    stem_image_data = np.interp(stem_image_data, [min, max], [0, 256])
    stem_image_data = stem_image_data.astype(np.uint8)
    img = Image.fromarray(stem_image_data)
    img.save(name)

with h5py.File('stem_image.h5', 'r') as rf:
    frames = rf['/electron_events/frames'][()]
    attrs = rf['/electron_events/frames'].attrs
    frame_width = attrs['Nx']
    frame_height = attrs['Ny']

    attrs = rf['/electron_events/scan_positions'].attrs
    scan_width = attrs['Nx']
    scan_height = attrs['Ny']

num_pixels = frame_width * frame_height

inner_radius = 40
outer_radius = 288

img = image.create_stem_image_sparse(frames, inner_radius, outer_radius,
                                     scan_width, scan_height, frame_width,
                                     frame_height)

save_img(img.bright, 'bright.png', scan_width, scan_height)
save_img(img.dark, 'dark.png', scan_width, scan_height)
