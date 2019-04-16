import click
import glob
from stempy import io, image
import sys
import time

# Beware of disk caching when using this benchmark
@click.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False))
@click.option('-n', '--num-runs', default=10, help='Number of runs to perform')
def run_benchmarks(path, num_runs):
    """Run benchmarks on a directory containing data*.dat files"""
    files = []
    for f in glob.glob(path + '/data*.dat'):
        files.append(f)

    if len(files) == 0:
        sys.exit('No data*.dat files found at: ' + path)

    times = []
    for i in range(num_runs):
        start = time.time()

        reader = io.reader(files)
        img = image.create_stem_image(reader, 160, 160,  40, 288);

        end = time.time()
        times.append(end - start)
        print('Run ' + str(len(times)) + ': {:0.2f} seconds'.format(times[-1]))

    print('Number of runs was:', num_runs)
    print('Average time: {:0.2f} seconds'.format(sum(times) / num_runs))


if __name__ == '__main__':
    run_benchmarks()
