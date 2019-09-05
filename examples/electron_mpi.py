from stempy import io, image

import click
import glob
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
from pathlib import Path
import sys

def get_files(files):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    return files[rank::world_size]


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
comm.Barrier()
start = MPI.Wtime()

# Every process will do the dark field reference average for now
dark_file = '/data/scan_62/data_scan62_dst0_file0.data'
reader = io.reader(dark_file, version=io.FileVersion.VERSION3)
dark = image.calculate_average(reader)

all_files = glob.glob('/data/scan_30/*.data')
files = get_files(all_files)

# Create local electron count
reader = io.reader(files, version=io.FileVersion.VERSION3)
electron_counted_data = image.electron_count(reader, dark, verbose=True)
local_frame_events = electron_counted_data.data

if rank == 0:
    global_frame_events = local_frame_events
else:
    global_frame_events = None

# Now reduce to root
if world_size > 1:
    if rank == 0:
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

    fig,ax=plt.subplots(figsize=(12,12))
    ax.matshow(img)
    plt.savefig('electron.png', dpi=300)

    # Save out the sparse image
    width = electron_counted_data.scan_width
    height = electron_counted_data.scan_height
    frame_width = electron_counted_data.frame_width
    frame_height = electron_counted_data.frame_height

    inner_radius = 0
    outer_radius = 50
    center_x = 286
    center_y = 314
    stem_img = image.create_stem_image_sparse(global_frame_events,
                                              inner_radius, outer_radius,
                                              width=width, height=height,
                                              frame_width=frame_width,
                                              frame_height=frame_height,
                                              center_x=center_x,
                                              center_y=center_y)

    fig,ax=plt.subplots(figsize=(12,12))
    ax.matshow(stem_img)
    plt.savefig('sparse_stem_image.png', dpi=300)
