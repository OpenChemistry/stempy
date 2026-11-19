import os
import tempfile

import h5py
import numpy as np
import pytest

from stempy import image, io


def test_simple():
    # Need at least 32 blocks since the blocksize in
    # io.get_hdf5_reader() is hard-coded to that.
    shape = (32, 576, 576)
    data = np.arange(np.prod(shape)).reshape(shape)
    dataset_name = "frames"

    # Stempy uses uint16. Let's prevent overflow...
    data %= np.iinfo(np.uint16).max
    mean = data.mean(axis=0)

    # Fake stuff that io.get_hdf5_reader() needs...
    fake_scan_name = "stem/images"
    fake_scan = np.zeros((4, 8, 1))

    temp = tempfile.NamedTemporaryFile(delete=False)
    file_name = temp.name
    try:
        with h5py.File(file_name, "w") as wf:
            wf.create_dataset(dataset_name, data=data)
            wf.create_dataset(fake_scan_name, data=fake_scan)

        with h5py.File(file_name, "r") as rf:
            reader = io.get_hdf5_reader(rf)
            average = image.calculate_average(reader)
    finally:
        del temp
        if os.path.exists(file_name):
            os.remove(file_name)

    # These should be equal...
    assert np.array_equal(average, mean)


def test_electron_count_user_thresholds():
    # Mostly empty frames can't be used for a Gaussian fit of the noise, so
    # this only works if the provided thresholds are used directly.
    frame_shape = (32, 32)
    shape = (32, *frame_shape)
    scan_dimensions = (4, 8)  # (width, height)
    data = np.zeros(shape, dtype=np.uint16)

    # One electron at the same pixel in every frame, plus an x-ray in frame 3
    data[:, 5, 7] = 100
    data[3, 10, 11] = 5000

    temp = tempfile.NamedTemporaryFile(delete=False)
    file_name = temp.name
    try:
        with h5py.File(file_name, "w") as wf:
            dset = wf.create_dataset("frames", data=data)
            dset.attrs["scan_dimensions"] = scan_dimensions
            wf.create_dataset("stem/images", data=np.zeros((4, 8, 1)))

        with h5py.File(file_name, "r") as rf:
            reader = io.get_hdf5_reader(rf)
            with pytest.warns(DeprecationWarning):
                events = image.electron_count(
                    reader,
                    darkreference=np.zeros(frame_shape),
                    scan_dimensions=scan_dimensions,
                    background_threshold=50,
                    xray_threshold=1000,
                )
    finally:
        del temp
        if os.path.exists(file_name):
            os.remove(file_name)

    assert events.data.shape[0] == np.prod(scan_dimensions)
    for frame_events in events.data[:, 0]:
        assert list(frame_events) == [5 * frame_shape[1] + 7]


def _write_sector_files(directory, scan_dimensions=(4, 4)):
    # Write version 5 sector files (one per module) with Gaussian noise and a
    # single electron in the same pixel of every sector.
    rng = np.random.default_rng(0)
    sector_size = 144 * 576
    dtype = np.dtype(
        [
            ("scan_num", "<u4"),
            ("frame_num", "<u4"),
            ("scan_size", "2<u2"),
            ("scan_pos", "2<u2"),
            ("data", f"{sector_size}<u2"),
        ]
    )
    num_frames = scan_dimensions[0] * scan_dimensions[1]
    paths = []
    for module in range(4):
        records = np.zeros(num_frames, dtype=dtype)
        for i in range(num_frames):
            records[i]["frame_num"] = i
            records[i]["scan_size"] = scan_dimensions
            records[i]["scan_pos"] = (i % scan_dimensions[0], i // scan_dimensions[0])
            sector = np.clip(rng.normal(100, 5, sector_size), 0, None)
            sector[1000] = 300
            records[i]["data"] = sector.astype("<u2")
        path = directory / f"data_scan1_module{module}_0.data"
        records.tofile(path)
        paths.append(str(path))

    return paths


def test_electron_count_user_thresholds_threaded_reader(tmp_path):
    scan_dimensions = (4, 4)
    paths = _write_sector_files(tmp_path, scan_dimensions)
    dark = np.zeros((576, 576), dtype=np.float32)

    def count(**kwargs):
        reader = io.reader(paths, io.FileVersion.VERSION5, backend="multi-pass")
        return image.electron_count(reader, dark, threshold_num_blocks=2, scan_dimensions=scan_dimensions, **kwargs)

    # The provided thresholds are used as is, and the fit is skipped.
    counted = count(background_threshold=200, xray_threshold=100000)
    metadata = counted.metadata["electron_counting"]
    assert metadata["background_threshold"] == 200
    assert metadata["x_ray_threshold"] == 100000
    assert not metadata["threshold_calculated"]
    # One electron in each of the 4 sectors of every frame
    assert all(len(frame_events) == 4 for frame_events in counted.data[:, 0])

    # Only one provided: the other is still calculated from the data
    counted = count(xray_threshold=50)
    metadata = counted.metadata["electron_counting"]
    assert metadata["x_ray_threshold"] == 50
    assert 100 < metadata["background_threshold"] < 150
    assert metadata["threshold_calculated"]

    # Nothing provided: both calculated. The x-ray threshold is well below the
    # electron intensity, so the electrons are excluded.
    counted = count()
    metadata = counted.metadata["electron_counting"]
    assert 100 < metadata["x_ray_threshold"] < 300
    assert metadata["threshold_calculated"]


if __name__ == "__main__":
    test_simple()
