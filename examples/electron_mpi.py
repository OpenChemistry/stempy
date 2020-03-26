from stempy import io, image

import click
from collections import namedtuple
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
@click.option('-c', '--center', help='The center (comma separated)',
              required=True)
@click.option('-i', '--inner-radii', help='The inner radii (comma separated)',
              required=True)
@click.option('-o', '--outer-radii', help='The outer radii (comma separated)',
              required=True)
@click.option('-f', '--output-file', help='The output HDF5 file to write',
              default='electron_counted_data.h5')
@click.option('-g', '--generate-sparse', is_flag=True,
              help='Generate and save sparse STEM image')
def main(files, dark_file, center, inner_radii, outer_radii, output_file,
         generate_sparse):
    center = center.split(',')
    if len(center) != 2:
        msg = 'Center must be of the form: center_x,center_y.'
        raise click.ClickException(msg)

    center = tuple(int(x) for x in center)

    inner_radii = inner_radii.split(',')
    outer_radii = outer_radii.split(',')

    if len(inner_radii) != len(outer_radii):
        msg = 'Number of inner and outer radii must match'
        raise click.ClickException(msg)

    inner_radii = [int(x) for x in inner_radii]
    outer_radii = [int(x) for x in outer_radii]

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

    if dark_file is not None:
        # Every process will do the dark field reference average for now
        reader = io.reader(dark_file, version=io.FileVersion.VERSION3)
        dark = image.calculate_average(reader)
    else:
        dark = np.zeros((576, 576))

    # Split up the files among processes
    files = get_files(files)

    # Create local electron count
    reader = io.reader(files, version=io.FileVersion.VERSION3)
    electron_counted_data = image.electron_count(reader, dark, verbose=True)
    local_frame_events = electron_counted_data.data

    # Now reduce to root
    global_frame_events = reduce_to_root_method1(local_frame_events)
    # global_frame_events = reduce_to_root_method2(local_frame_events)

    comm.Barrier()
    end = MPI.Wtime()

    if rank == 0:
        print('time: %s' % (end - start))

    if rank == 0:
        # Create new electron counted data with the global frame events
        data = namedtuple('ElectronCountedData',
                          ['data', 'scan_dimensions', 'frame_dimensions'])
        data.data = global_frame_events
        data.scan_dimensions = electron_counted_data.scan_dimensions
        data.frame_dimensions = electron_counted_data.frame_dimensions

        # Write out the HDF5 file
        io.save_electron_counts(output_file, data)

        if generate_sparse:
            # Save out the sparse image

            stem_imgs = image.create_stem_images(data, inner_radii,
                                                 outer_radii, center=center)

            for i, img in enumerate(stem_imgs):
                fig, ax = plt.subplots(figsize=(12, 12))
                ax.matshow(img)
                name = 'sparse_stem_image_' + str(i) + '.png'
                plt.savefig(name, dpi=300)


def reduce_to_root_method1(local_frame_events):
    # This method uses send() and recv() with the data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_size == 1:
        return local_frame_events

    global_frame_events = None
    if rank == 0:
        global_frame_events = local_frame_events
        for i in range(1, world_size):
            data = comm.recv(source=i)
            for j in range(data.shape[0]):
                if len(data[j]) != 0:
                    global_frame_events[j] = data[j]
    else:
        comm.send(local_frame_events, dest=0)

    return global_frame_events


def reduce_to_root_method2(local_frame_events):
    # This method uses comm.reduce() with the data
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_size == 1:
        return local_frame_events

    # Must replace empty arrays with np.array([0]) for sum to work
    for i in range(local_frame_events.shape[0]):
        if len(local_frame_events[i]) == 0:
            local_frame_events[i] = np.array([0])

    return comm.reduce(local_frame_events, op=MPI.SUM)


def reduce_to_root_method3(local_frame_events):
    # This method uses comm.Reduce() with the data
    # We may want to try comm.Reduce() sometime, but we will need
    # to come up with a way to convert our array of variable length
    # numpy arrays into a contiguous memory buffer.
    pass


if __name__ == "__main__":
    main()
