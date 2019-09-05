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
@click.option('-d', '--dark-file', help='The file for dark field reference',
              required=True)
@click.option('-c', '--center', help='The center (comma separated)',
              required=True)
@click.option('-i', '--inner-radii', help='The inner radii (comma separated)',
              required=True)
@click.option('-o', '--outer-radii', help='The outer radii (comma separated)',
              required=True)
@click.option('-g', '--generate-sparse',
              help='Generate and save sparse STEM image', default=False)
def main(files, dark_file, center, inner_radii, outer_radii, generate_sparse):
    center = center.split(',')
    if len(center) != 2:
        msg = 'Center must be of the form: center_x,center_y.'
        raise click.ClickException(msg)

    center_x, center_y = [int(x) for x in center]

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
    comm.Barrier()
    start = MPI.Wtime()

    # Every process will do the dark field reference average for now
    reader = io.reader(dark_file, version=io.FileVersion.VERSION3)
    dark = image.calculate_average(reader)

    # Split up the files among processes
    files = get_files(files)

    # Create local electron count
    reader = io.reader(files, version=io.FileVersion.VERSION3)
    electron_counted_data = image.electron_count(reader, dark, verbose=True)
    local_frame_events = electron_counted_data.data

    # Now reduce to root
    if world_size > 1:
        if rank == 0:
            global_frame_events = local_frame_events
            # If comm.reduce() is used here instead, the empty numpy arrays
            # in local_frame_events need to be replaced with np.array([0])
            # so that they can be summed.
            # Using comm.reduce() was found to take just a little longer on
            # ulex than the method below.
            # We may want to try comm.Reduce() sometime, but we will need
            # to come up with a way to convert our array of variable length
            # numpy arrays into a contiguous memory buffer.
            for i in range(1, world_size):
                data = comm.recv(source=i)
                for j in range(data.shape[0]):
                    if len(data[j]) != 0:
                        global_frame_events[j] = data[j]
        else:
            comm.send(local_frame_events, dest=0)
    else:
        global_frame_events = local_frame_events

    comm.Barrier()
    end = MPI.Wtime()

    if rank == 0:
        print('time: %s' % (end - start))

    if rank == 0:
        # Save out the electron counted image
        img = np.zeros((576, 576))

        # Just sum the events for now
        for frame in global_frame_events:
            for pos in frame:
                row = pos // 576
                column = pos % 576
                img[row][column] += 1

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.matshow(img)
        plt.savefig('electron.png', dpi=300)

        if generate_sparse:
            # Save out the sparse image
            width = electron_counted_data.scan_width
            height = electron_counted_data.scan_height
            frame_width = electron_counted_data.frame_width
            frame_height = electron_counted_data.frame_height

            stem_img = image.create_stem_images_sparse(
                global_frame_events, inner_radii, outer_radii, width=width,
                height=height, frame_width=frame_width,
                frame_height=frame_height, center_x=center_x,
                center_y=center_y)

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.matshow(stem_img)
            plt.savefig('sparse_stem_image.png', dpi=300)


if __name__ == "__main__":
    main()
