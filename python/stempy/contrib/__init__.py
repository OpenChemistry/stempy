
def get_scan_path(directory, scan_num=None, scan_id=None, th=None):
    """
    Get the file path for a 4D Camera scan from a directory on NERSC
    using the scan number, the Distiller scan id, and/or threshold.
    scan_id should always be unique and is the best option to load a dataset.
    
    A ValueError is raised if more than one file matches the input. Then the user
    needs to input more information to narrow down the choices.

    :param directory: The path to the directory containing the file.
    :type path: pathlib.Path or str
    :param scan_num: The 4D Camera scan number. Optional,
    :type scan_num: int, optional
    :param scan_id: The Distiller scan id.
    :type scan_id: int, optional
    :param th: The threshold for counting. This was added to the filename in older files. Optional.
    :type th: int, optional
    :return: A tuple containing the file Path that matches the input information, the scan_num, and the scan_id.
    :rtype: (pathlib.Path, int, int)
    """
    
    if scan_id is not None:
        # This should be unique
        file_path = list(directory.glob(f'*_id{scan_id}*.h5'))
    elif scan_num is not None:
        # older files might include the threshold (th)
        if th is not None:
            file_path = list(directory.glob(f'data_scan{scan_num}_th{th}_electrons.h5'))
        else:
            file_path = list(directory.glob(f'data_scan{scan_num}*electrons.h5'))
    else:
        raise TypeError('Missing scan_num or scan_id input.')

    if len(file_path) > 1:
        raise ValueError('Multiple files match that input. Add scan_id to be more specific.')
    elif len(file_path) == 1:
        file_path = file_path[0]
        # Determine the scan_id and scan_num for use later (i.e. getting DM4 file)
        spl = file_path.name.split('_')
        for ii in spl:
            if 'id' in ii:
                scan_id = int(ii[len('id'):])
            elif 'scan' in ii:
                scan_num = int(ii[len('scan'):])
    else:
        raise FileNotFoundError('No file with those parameters can be found.')
    return file_path, scan_num, scan_id
