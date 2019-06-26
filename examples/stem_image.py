import glob
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

stem_image_data_day = np.zeros((160*160,), dtype=float)
stem_image_data_night = np.zeros((160*160,), dtype=float)

mask_size = 20
files = []
for f in glob.glob('/data/4dstem/smallScanningDiffraction/data*.dat'):
    files.append(f)

inner_radii = [0, 40]
outer_radii = [288, 288]

reader = io.reader(files)
imgs = image.create_stem_images(reader, inner_radii, outer_radii, width=160,
                                height=160)

for i, img in enumerate(imgs):
    suffix = str(inner_radii[i]) + '_' + str(outer_radii[i]) + '.png'
    save_img(img, 'img_' + suffix)
