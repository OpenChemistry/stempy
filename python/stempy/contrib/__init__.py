from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


class FileSuffix(Enum):
    STANDARD = ""
    OFFSETS = "_offsets"
    CENTERED = "_centered"


suffix_to_filetype = {
    FileSuffix.STANDARD: "h5",
    FileSuffix.OFFSETS: "emd",
    FileSuffix.CENTERED: "h5",
}


def check_only_one_filepath(file_path):
    if len(file_path) > 1:
        raise ValueError(
            "Multiple files match that input. Add scan_id to be more specific."
        )
    elif len(file_path) == 0:
        raise FileNotFoundError("No file with those parameters can be found.")


def get_scan_path_version_0(
    directory: Path,
    scan_num: Optional[int] = None,
    scan_id: Optional[int] = None,
    th: Optional[int] = None,
    file_suffix: FileSuffix = FileSuffix.STANDARD,
) -> Tuple[Path, Optional[int], Optional[int]]:
    """
    Filename looks like: data_scan{scan_num}_id{scan_id}_electrons.h5

    Parameters
    ----------
    directory : pathlib.Path or str
        The path to the directory containing the file.
    scan_id : int, optional
        The Distiller scan id.
    scan_num : int, optional
        The 4D Camera scan number.
    th : int, optional
        The threshold number used to name the file.

    Returns
    -------
    pathlib.Path
        The path to the found file.
    int
        The scan number extracted from the file name.
    int
        The scan id extracted from the file name.
    """

    if scan_id is None and scan_num is None:
        raise TypeError("Either scan_num or scan_id must be provided.")

    filetype = suffix_to_filetype[file_suffix]

    file_pattern = ""
    if scan_id is not None:
        file_pattern = f"*_id{scan_id}*{file_suffix.value}.{filetype}"
    elif scan_num is not None:
        file_pattern = (
            f"data_scan{scan_num}_th{th}_electrons{file_suffix.value}.{filetype}"
            if th is not None
            else f"data_scan{scan_num}*electrons{file_suffix.value}.{filetype}"
        )

    file_path = list(directory.glob(file_pattern))

    check_only_one_filepath(file_path)

    file_path = file_path[0]
    spl = file_path.name.split("_")
    for part in spl:
        if "id" in part:
            scan_id = int(part[2:])
        elif "scan" in part:
            scan_num = int(part[4:])

    return file_path, scan_num, scan_id


def get_scan_path_version_1(
    directory: Path,
    scan_num: Optional[int] = None,
    scan_id: Optional[int] = None,
    file_suffix: FileSuffix = FileSuffix.STANDARD,
) -> Tuple[Path, Optional[int], Optional[int]]:
    """
    File name looks like: FOURD_YYMMDD_HHMM_SSSSS_NNNNN.h5
    where YYMMDD is the date, HHMM is the time,
    SSSSS is the scan_id, and NNNNN is the scan_num.

    Parameters
    ----------
    directory : pathlib.Path or str
        The path to the directory containing the file.
    scan_id : int, optional
        The Distiller scan id.
    scan_num : int, optional
        The 4D Camera scan number.

    Returns
    -------
    pathlib.Path
        The path to the found file.
    int
        The scan number extracted from the file name.
    int
        The scan id extracted from the file name.
    """

    if scan_id is None and scan_num is None:
        raise TypeError("Either scan_num or scan_id must be provided.")

    file_pattern = ""
    filetype = suffix_to_filetype[file_suffix]
    if scan_id is not None:
        if file_suffix == FileSuffix.STANDARD:
            file_pattern = f"FOURD_*_{str(scan_id).zfill(5)}_?????.{filetype}"
        else:
            file_pattern = (
                f"FOURD_*_{str(scan_id).zfill(5)}_?????{file_suffix.value}.{filetype}"
            )
    elif scan_num is not None:
        file_pattern = (
            f"FOURD_*_*_{str(scan_num).zfill(5)}{file_suffix.value}.{filetype}"
        )

    file_path = list(directory.glob(file_pattern))
    check_only_one_filepath(file_path)
    file_path = file_path[0]
    spl = file_path.stem.split("_")
    scan_id = int(spl[3])
    scan_num = int(spl[4])

    return file_path, scan_num, scan_id


def get_scan_path(
    directory: Path,
    scan_num: Optional[int] = None,
    scan_id: Optional[int] = None,
    th: Optional[int] = None,
    file_suffix: FileSuffix = FileSuffix.STANDARD,
) -> Tuple[Path, Optional[int], Optional[int]]:
    """Get the file path for a 4D Camera scan on NERSC using the scan number,
    the Distiller scan id, and/or threshold. scan_id should always
    be unique and is the best option to load a dataset.

    A ValueError is raised if more than one file matches the input. Then the user
    needs to input more information to narrow down the choices.

    Parameters
    ----------
    directory : pathlib.Path or str
        The path to the directory containing the file.
    scan_id : int, optional
        The Distiller scan id.
    scan_num : int, optional
        The 4D Camera scan number. Optional
    th : float, optional
        The threshold for counting. This was added to the filename in older files.

    Returns
    -------
    : tuple
        The tuple contains the file that matches the input information and the
        scan_num and scan_id as a tuple.
    """
    try:
        return get_scan_path_version_1(
            directory, scan_num, scan_id, file_suffix=file_suffix
        )
    except FileNotFoundError:
        return get_scan_path_version_0(
            directory,
            scan_num=scan_num,
            scan_id=scan_id,
            th=th,
            file_suffix=file_suffix,
        )
