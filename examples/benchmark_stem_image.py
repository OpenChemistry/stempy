import glob
from stempy import io, image
import time

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
