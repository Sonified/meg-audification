"""MEG data loading utilities supporting multiple formats."""

from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import numpy as np


def load_meg_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load MEG data from various formats.

    Supports:
        - .fif (MNE-Python native format)
        - .mat (MATLAB format, both v7 and v7.3+)
        - .csv (CSV with time column + data columns)

    Parameters
    ----------
    filepath : str or Path
        Path to the MEG data file.

    Returns
    -------
    dict
        Dictionary containing:
        - 'data': numpy array of shape (n_channels, n_samples)
        - 'sfreq': sampling frequency in Hz
        - 'ch_names': list of channel names (if available)
        - 'format': original file format ('fif', 'mat')
        - 'raw_obj': original MNE Raw object (only for .fif files)
        - 'mat_contents': full MATLAB file contents (only for .mat files)

    Raises
    ------
    ValueError
        If file format is not supported.
    FileNotFoundError
        If file does not exist.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == '.fif' or filepath.name.endswith('.fif.gz'):
        return _load_fif(filepath)
    elif suffix == '.mat':
        return _load_mat(filepath)
    elif suffix == '.csv':
        return _load_csv(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .fif, .fif.gz, .mat, .csv"
        )


def _load_fif(filepath: Path) -> Dict[str, Any]:
    """Load MNE-Python .fif format."""
    import mne

    raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    data = raw.get_data()

    return {
        'data': data,
        'sfreq': raw.info['sfreq'],
        'ch_names': raw.ch_names,
        'format': 'fif',
        'raw_obj': raw,
        'mat_contents': None,
    }


def _load_csv(filepath: Path) -> Dict[str, Any]:
    """
    Load CSV format MEG data.

    Expects format: time_column, data_column1, data_column2, ...
    Infers sampling frequency from time column.
    """
    # Load CSV data
    data = np.loadtxt(filepath, delimiter=',')

    # First column is time, remaining columns are data channels
    time = data[:, 0]
    channels = data[:, 1:].T  # Transpose to (n_channels, n_samples)

    # Infer sampling frequency from time column
    if len(time) > 1:
        dt = time[1] - time[0]
        sfreq = 1.0 / dt
    else:
        sfreq = None

    # Generate channel names
    n_channels = channels.shape[0]
    ch_names = [f'CH{i+1}' for i in range(n_channels)]

    return {
        'data': channels,
        'sfreq': sfreq,
        'ch_names': ch_names,
        'time': time,
        'format': 'csv',
        'raw_obj': None,
        'mat_contents': None,
    }


def _load_mat(filepath: Path) -> Dict[str, Any]:
    """
    Load MATLAB .mat format.

    Handles both MATLAB v7 (scipy) and v7.3+ (h5py) formats.
    """
    mat_contents = _load_mat_file(filepath)

    # Try to extract MEG data - this will need refinement once we see the actual data
    data, sfreq, ch_names = _extract_meg_from_mat(mat_contents)

    return {
        'data': data,
        'sfreq': sfreq,
        'ch_names': ch_names,
        'format': 'mat',
        'raw_obj': None,
        'mat_contents': mat_contents,
    }


def _load_mat_file(filepath: Path) -> Dict[str, Any]:
    """
    Load .mat file, automatically detecting version.

    MATLAB v7.3+ files use HDF5 format and require h5py.
    Older versions use scipy.io.loadmat.
    """
    # First try scipy (works for v7 and earlier)
    try:
        from scipy.io import loadmat
        contents = loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
        # Remove MATLAB metadata keys
        contents = {k: v for k, v in contents.items() if not k.startswith('__')}
        return contents
    except NotImplementedError:
        # v7.3+ file, need h5py
        pass

    # Try h5py for v7.3+ (HDF5-based) files
    import h5py

    contents = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            contents[key] = _h5py_to_numpy(f[key])

    return contents


def _h5py_to_numpy(item) -> Any:
    """Recursively convert h5py objects to numpy arrays/dicts."""
    import h5py

    if isinstance(item, h5py.Dataset):
        val = item[()]
        # Handle MATLAB strings stored as uint16
        if val.dtype == np.uint16:
            try:
                return ''.join(chr(c) for c in val.flatten())
            except (ValueError, TypeError):
                return val
        return val
    elif isinstance(item, h5py.Group):
        return {k: _h5py_to_numpy(v) for k, v in item.items()}
    else:
        return item


def _extract_meg_from_mat(mat_contents: Dict[str, Any]) -> Tuple[np.ndarray, float, Optional[list]]:
    """
    Extract MEG data from MATLAB file contents.

    This function attempts to find MEG data in common structures.
    It will need to be updated once we see the actual data format.

    Common field names to look for:
        - FieldTrip: 'data.trial', 'data.time', 'data.fsample', 'data.label'
        - Brainstorm: 'F', 'Time', 'Channel'
        - Generic: 'data', 'Data', 'meg', 'MEG', 'eeg', 'EEG'
    """
    data = None
    sfreq = None
    ch_names = None

    # Try FieldTrip format
    if 'data' in mat_contents or 'Data' in mat_contents:
        ft_data = mat_contents.get('data', mat_contents.get('Data'))
        if hasattr(ft_data, 'trial'):
            # FieldTrip structure
            data = np.array(ft_data.trial)
            if data.ndim == 1:  # Single trial stored as object array
                data = data[0]
            sfreq = float(ft_data.fsample) if hasattr(ft_data, 'fsample') else None
            ch_names = list(ft_data.label) if hasattr(ft_data, 'label') else None
        elif isinstance(ft_data, np.ndarray):
            data = ft_data

    # Try Brainstorm format
    if data is None and 'F' in mat_contents:
        data = mat_contents['F']
        if 'Time' in mat_contents:
            time = mat_contents['Time']
            if len(time) > 1:
                sfreq = 1.0 / (time[1] - time[0])

    # Try generic field names
    if data is None:
        for key in ['meg', 'MEG', 'eeg', 'EEG', 'raw', 'Raw', 'signals', 'Signals']:
            if key in mat_contents:
                candidate = mat_contents[key]
                if isinstance(candidate, np.ndarray) and candidate.ndim == 2:
                    data = candidate
                    break

    # Try to find sampling frequency if not yet found
    if sfreq is None:
        for key in ['sfreq', 'Fs', 'fs', 'fsample', 'SampleRate', 'srate', 'sr']:
            if key in mat_contents:
                sfreq = float(mat_contents[key])
                break

    # Try to find channel names if not yet found
    if ch_names is None:
        for key in ['ch_names', 'channels', 'Channels', 'label', 'labels', 'chanlocs']:
            if key in mat_contents:
                candidate = mat_contents[key]
                if isinstance(candidate, (list, np.ndarray)):
                    ch_names = [str(c) for c in candidate]
                    break

    if data is None:
        # Return the keys so user can identify the structure
        available_keys = list(mat_contents.keys())
        raise ValueError(
            f"Could not automatically extract MEG data from .mat file. "
            f"Available keys: {available_keys}. "
            f"Please inspect mat_contents and update _extract_meg_from_mat()."
        )

    # Ensure data is 2D (channels x samples)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data, sfreq, ch_names


def inspect_mat_file(filepath: Union[str, Path]) -> Dict[str, str]:
    """
    Inspect a MATLAB file and return information about its structure.

    Useful for understanding the data format before loading.

    Parameters
    ----------
    filepath : str or Path
        Path to the .mat file.

    Returns
    -------
    dict
        Dictionary mapping field names to descriptions of their contents.
    """
    filepath = Path(filepath)
    mat_contents = _load_mat_file(filepath)

    info = {}
    for key, value in mat_contents.items():
        if isinstance(value, np.ndarray):
            info[key] = f"ndarray, shape={value.shape}, dtype={value.dtype}"
        elif hasattr(value, '__dict__'):
            attrs = list(value.__dict__.keys()) if hasattr(value, '__dict__') else []
            info[key] = f"struct with attributes: {attrs}"
        else:
            info[key] = f"{type(value).__name__}"

    return info
