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

reader = io.reader(files)
img = image.create_stem_image(reader, 160, 160,  40, 288);

save_img(img.bright, 'bright.png')
save_img(img.dark, 'dark.png')
