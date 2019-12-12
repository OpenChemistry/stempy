from stempy import io, image

import click
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np


def get_files(files):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    return files[rank::world_size]


@click.command()
@click.argument('files', nargs=-1,
                type=click.Path(exists=True, dir_okay=False))
@click.option('-d', '--dark-file', help='The file for dark field reference')
@click.option('-o', '--output-file', help='The output npz file to write',
              default='mdp.npy')
def main(files, dark_file, output_file):
    """
    Example of calculating maximum diffracton pattern using MPI.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if (world_size > len(files)):
        if rank == 0:
            print('Error: number of MPI processes,', world_size, ', exceeds',
                  'the number of files:', len(files))
        return

    comm.Barrier()
    start = MPI.Wtime()

    # Split up the files among processes
    files = get_files(files)

    # Create local maximum diffraction pattern
    reader = io.reader(files, version=io.FileVersion.VERSION3)
    mdp = image.maximum_diffraction_pattern(reader)
    mdp = mdp.data

    # Now reduce to root
    global_mdp = np.zeros_like(mdp)
    comm.Reduce(mdp, global_mdp, op=MPI.MAX)

    if dark_file is not None:
        reader = io.reader(dark_file, version=io.FileVersion.VERSION3)
        dark = image.calculate_average(reader)
        mdp -= dark

    comm.Barrier()
    end = MPI.Wtime()

    if rank == 0:
        print('time: %s' % (end - start))

    if rank == 0:
        # Write out the MDP
        np.save(output_file, global_mdp)


if __name__ == "__main__":
    main()
