import click
from stempy import io, image
import subprocess
import sys
import time

# Beware of disk caching when using this benchmark
@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-d', '--dark-sample', help='Dark field sample',
              type=click.Path(exists=True, dir_okay=False))
@click.option('-n', '--num-runs', default=10, help='Number of runs to perform')
def run_benchmarks(files, dark_sample, num_runs):
    """Run benchmarks using the files given in the arguments

    Example: "python benchmark_stem_image.py -d darksample.dat /path/to/data/data*.dat"

    """

    if len(files) == 0:
        sys.exit('No files found')

    times = []
    for i in range(num_runs):
        # You can use a command like this to clear the caches, where
        # `clearcaches` should be a bash script.
        # subprocess.Popen('clearcaches').wait()
        start = time.time()

        reader = io.reader(dark_sample, version=io.FileVersion.VERSION2)
        dark = image.calculate_average(reader)

        reader = io.reader(files, version=io.FileVersion.VERSION2)
        frame_events = image.electron_count(reader, dark, scan_width=40,
                                            scan_height=40)

        end = time.time()
        times.append(end - start)
        print('Run ' + str(len(times)) + ': {:0.2f} seconds'.format(times[-1]))

    print('Number of runs was:', num_runs)
    print('Average time: {:0.2f} seconds'.format(sum(times) / num_runs))

if __name__ == '__main__':
    run_benchmarks()
