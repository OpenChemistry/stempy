import glob
from stempy import io, image
import numpy as np
from PIL import Image
import time

def save_img(stem_image_data, name):
    min = np.min(stem_image_data)
    max = np.max(stem_image_data)

    stem_image_data = stem_image_data.reshape((160, 160))
    stem_image_data = np.interp(stem_image_data, [min, max], [0, 256])
    stem_image_data = stem_image_data.astype(np.uint8)
    img = Image.fromarray(stem_image_data)
    img.save(name)

# Beware of disk caching when using this benchmark
num_runs = 10
times = []
for i in range(num_runs):
    start = time.time()

    files = []
    for f in glob.glob('/data/4dstem/smallScanningDiffraction/data*.dat'):
        files.append(f)

    reader = io.reader(files)
    img = image.create_stem_image(reader, 160, 160,  40, 288);

    end = time.time()
    times.append(end - start)
    print('Run ' + str(len(times)) + ': {:0.2f} seconds'.format(times[-1]))

print('Number of runs was:', num_runs)
print('Average time: {:0.2f} seconds'.format(sum(times) / num_runs))

save_img(img.bright, 'bright.png')
save_img(img.dark, 'dark.png')
