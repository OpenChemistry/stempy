import click
from stempy import io, image
import sys
import time

# Beware of disk caching when using this benchmark
@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-n', '--num-runs', default=10, help='Number of runs to perform')
def run_benchmarks(files, num_runs):
    """Run benchmarks using the files given in the arguments

    Example: "python benchmark_stem_image.py /path/to/data/data*.dat"

    """

    if len(files) == 0:
        sys.exit('No files found')

    times = []
    for i in range(num_runs):
        start = time.time()

        reader = io.reader(files)
        img = image.create_stem_image(reader, 40, 288,
                                      scan_dimensions=(160, 160))

        end = time.time()
        times.append(end - start)
        print('Run ' + str(len(times)) + ': {:0.2f} seconds'.format(times[-1]))

    print('Number of runs was:', num_runs)
    print('Average time: {:0.2f} seconds'.format(sum(times) / num_runs))


if __name__ == '__main__':
    run_benchmarks()
