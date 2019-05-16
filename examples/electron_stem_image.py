import glob
import h5py
from stempy import io, image
import numpy as np
from PIL import Image

def save_img(stem_image_data, name, rows, columns):
    min = np.min(stem_image_data)
    max = np.max(stem_image_data)

    stem_image_data = stem_image_data.reshape((rows, columns))
    stem_image_data = np.interp(stem_image_data, [min, max], [0, 256])
    stem_image_data = stem_image_data.astype(np.uint8)
    img = Image.fromarray(stem_image_data)
    img.save(name)

with h5py.File('stem_image.h5', 'r') as rf:
    frames = rf['/electron_events/frames'][()]
    attrs = rf['/electron_events/frames'].attrs
    frame_rows = attrs['Ny']
    frame_columns = attrs['Nx']

    attrs = rf['/electron_events/scan_positions'].attrs
    rows = attrs['Ny']
    columns = attrs['Nx']

num_pixels = frame_rows * frame_columns

inner_radius = 40
outer_radius = 288

img = image.create_stem_image_sparse(frames, inner_radius, outer_radius,
                                     rows, columns, frame_rows, frame_columns)

save_img(img.bright, 'bright.png', rows, columns)
save_img(img.dark, 'dark.png', rows, columns)
