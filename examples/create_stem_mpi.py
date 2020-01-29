import glob
from stempy import io, image
import numpy as np
from mpi4py import MPI
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import click

def get_files(files):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    return files[rank::world_size]

@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-c', '--center', help='The center (comma separated)', required=True)
@click.option('-i', '--inner-radii', help='The inner radii (comma separated)', required=True)
@click.option('-o', '--outer-radii', help='The outer radii (comma separated)', required=True)
def main(files, center, inner_radii, outer_radii):
    center = center.split(',')
    if len(center) != 2:
        raise click.ClickException('Center must be of the form: center_x,center_y.')

    center = tuple(int(x) for x in center)

    inner_radii = inner_radii.split(',')
    outer_radii = outer_radii.split(',')

    if len(inner_radii) != len(outer_radii):
        raise click.ClickException('Number of inner and outer radii must match')

    inner_radii = [int(x) for x in inner_radii]
    outer_radii = [int(x) for x in outer_radii]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    comm.Barrier()
    start = MPI.Wtime()

    files = get_files(files)

    # Create local sum
    reader = io.reader(files, version=io.FileVersion.VERSION3)

    # Get the scan image size
    block = reader.read()
    scan_dimensions = block.header.scan_dimensions
    reader.reset()
    local_stems = image.create_stem_images(reader, inner_radii, outer_radii, center=center)

    # Now reduce to root
    global_stems = [np.zeros(scan_dimensions[0] * scan_dimensions[1], dtype='uint64') for _ in range(len(inner_radii))]
    for i in range(len(inner_radii)):
        comm.Reduce(local_stems[i], global_stems[i], op=MPI.SUM)

    # Save out the image
    if rank == 0:
        for global_stem, inner, outer in zip(global_stems, inner_radii, outer_radii):
            thr = global_stem[:] > 0
            vmin = global_stem[thr].min()

            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin=vmin)

            stem_image = cmap(norm(global_stem.reshape(scan_dimensions[1], scan_dimensions[0])))
            filename = 'stem_%d_%d.png' % (inner, outer)
            plt.imsave(filename, stem_image)

    comm.Barrier()
    end = MPI.Wtime()

    if rank == 0:
        print('time: %s' % (end - start))

if __name__ == "__main__":
    main()
