"""MEG signal to audio conversion utilities."""

from pathlib import Path
from typing import Union, Optional, Tuple, Literal
import numpy as np
from scipy.io import wavfile
from scipy import signal

# Default amplitude for normalization (0.0 to 1.0)
DEFAULT_TARGET_AMPLITUDE = 1.0


def normalize_data(
    data: np.ndarray,
    target_amplitude: float = DEFAULT_TARGET_AMPLITUDE,
    method: Literal['peak', 'rms'] = 'peak'
) -> np.ndarray:
    """
    Normalize audio data to a target amplitude.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array (1D or 2D).
    target_amplitude : float
        Target amplitude (0.0 to 1.0). Default DEFAULT_TARGET_AMPLITUDE.
    method : {'peak', 'rms'}
        'peak': Scale by maximum absolute value (default).
        'rms': Scale by RMS value (consistent loudness).

    Returns
    -------
    numpy.ndarray
        Normalized data scaled to [-target_amplitude, target_amplitude].
    """
    data = np.asarray(data, dtype=np.float64)

    if method == 'peak':
        max_amplitude = np.max(np.abs(data))
        if max_amplitude > 0:
            return data * (target_amplitude / max_amplitude)
        return data

    elif method == 'rms':
        rms = np.sqrt(np.mean(data ** 2))
        if rms > 0:
            # Scale RMS to ~0.2 (leaves headroom for peaks)
            target_rms = target_amplitude * 0.22
            return data * (target_rms / rms)
        return data

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_signal(
    data: np.ndarray,
    sfreq: float,
    remove_dc: bool = True,
    highpass_freq: Optional[float] = None,
    taper_percent: float = 0.01
) -> np.ndarray:
    """
    Preprocess signal before audification.

    Parameters
    ----------
    data : numpy.ndarray
        Input signal (1D: samples, or 2D: channels x samples).
    sfreq : float
        Sampling frequency in Hz.
    remove_dc : bool
        Remove DC offset (mean) from signal. Default True.
    highpass_freq : float, optional
        High-pass filter frequency in Hz. Removes slow drifts.
        If None, no filtering is applied.
    taper_percent : float
        Percentage of signal to taper at edges (0.0 to 0.5).
        Prevents click artifacts. Default 0.01 (1%).

    Returns
    -------
    numpy.ndarray
        Preprocessed signal.
    """
    data = np.asarray(data, dtype=np.float64)
    was_1d = data.ndim == 1
    if was_1d:
        data = data.reshape(1, -1)

    # Remove DC offset (commented out - may not be needed for MEG data)
    # if remove_dc:
    #     data = data - np.mean(data, axis=1, keepdims=True)

    # High-pass filter to remove slow drifts
    if highpass_freq is not None and highpass_freq > 0:
        nyquist = sfreq / 2
        if highpass_freq < nyquist:
            sos = signal.butter(4, highpass_freq, 'hp', fs=sfreq, output='sos')
            data = signal.sosfiltfilt(sos, data, axis=1)

    # Taper edges to prevent clicks
    if taper_percent > 0:
        n_samples = data.shape[1]
        taper_samples = int(n_samples * taper_percent)
        if taper_samples > 0:
            taper = np.ones(n_samples)
            # Cosine taper (Tukey window edges)
            taper[:taper_samples] = 0.5 * (1 - np.cos(np.pi * np.arange(taper_samples) / taper_samples))
            taper[-taper_samples:] = 0.5 * (1 - np.cos(np.pi * np.arange(taper_samples, 0, -1) / taper_samples))
            data = data * taper

    if was_1d:
        return data[0]
    return data


def meg_to_audio(
    data: np.ndarray,
    sfreq: float,
    output_sfreq: int = 44100,
    target_amplitude: float = DEFAULT_TARGET_AMPLITUDE,
    remove_dc: bool = False,
    highpass_freq: Optional[float] = 0.1,
    taper_percent: float = 0.01,
    channel_mix: Literal['mean', 'first', 'all'] = 'mean'
) -> Tuple[np.ndarray, int]:
    """
    Convert MEG signal(s) to audio-ready format.

    Parameters
    ----------
    data : numpy.ndarray
        MEG data. Shape: (n_channels, n_samples) or (n_samples,).
    sfreq : float
        Original sampling frequency in Hz.
    output_sfreq : int
        Target audio sampling rate. Default 44100 Hz.
    target_amplitude : float
        Target peak amplitude (0.0 to 1.0). Default DEFAULT_TARGET_AMPLITUDE.
    remove_dc : bool
        Remove DC offset before conversion. Default False.
    highpass_freq : float, optional
        High-pass filter frequency. Default 0.1 Hz.
    taper_percent : float
        Edge taper percentage. Default 0.01.
    channel_mix : {'mean', 'first', 'all'}
        How to handle multiple channels:
        - 'mean': Average all channels to mono (default).
        - 'first': Use only the first channel.
        - 'all': Keep all channels (returns stereo if 2, else mono from mean).

    Returns
    -------
    audio_data : numpy.ndarray
        Audio data as float64 in range [-1, 1].
        Shape: (n_samples,) for mono, (n_samples, 2) for stereo.
    output_sfreq : int
        Output sampling rate.
    """
    data = np.asarray(data, dtype=np.float64)

    # Ensure 2D (channels x samples)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Preprocess
    data = preprocess_signal(
        data, sfreq,
        remove_dc=remove_dc,
        highpass_freq=highpass_freq,
        taper_percent=taper_percent
    )

    # Mix channels
    if channel_mix == 'mean':
        audio = np.mean(data, axis=0)
    elif channel_mix == 'first':
        audio = data[0]
    elif channel_mix == 'all':
        if data.shape[0] == 2:
            audio = data.T  # (samples, 2) for stereo
        else:
            audio = np.mean(data, axis=0)
    else:
        raise ValueError(f"Unknown channel_mix: {channel_mix}")

    # No resampling - just set the playback rate
    # Playing 1000 Hz data at 44100 Hz shifts frequencies up by 44.1x
    # (e.g., 10 Hz alpha becomes ~441 Hz - now audible)

    # Normalize
    audio = normalize_data(audio, target_amplitude=target_amplitude)

    return audio, output_sfreq


def save_audio(
    audio_data: np.ndarray,
    sfreq: int,
    filepath: Union[str, Path],
    bit_depth: Literal[16, 24, 32] = 16
) -> Path:
    """
    Save audio data to WAV file.

    Parameters
    ----------
    audio_data : numpy.ndarray
        Audio data as float in range [-1, 1].
    sfreq : int
        Sampling rate in Hz.
    filepath : str or Path
        Output file path.
    bit_depth : {16, 24, 32}
        Bit depth for output. Default 16.

    Returns
    -------
    Path
        Path to saved file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to appropriate integer format
    if bit_depth == 16:
        max_val = 32767
        dtype = np.int16
    elif bit_depth == 24:
        # scipy.io.wavfile doesn't support 24-bit directly
        # Fall back to 32-bit float
        max_val = 1.0
        dtype = np.float32
    elif bit_depth == 32:
        max_val = 1.0
        dtype = np.float32
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    if dtype in (np.int16,):
        # Clip and convert to integer
        audio_int = np.clip(audio_data * max_val, -max_val, max_val).astype(dtype)
        wavfile.write(str(filepath), sfreq, audio_int)
    else:
        # Float format - ensure in [-1, 1]
        audio_float = np.clip(audio_data, -1.0, 1.0).astype(dtype)
        wavfile.write(str(filepath), sfreq, audio_float)

    return filepath


def audify_meg_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    output_sfreq: int = 44100,
    target_amplitude: float = DEFAULT_TARGET_AMPLITUDE,
    channel_indices: Optional[list] = None,
    channel_mix: Literal['mean', 'first', 'all'] = 'mean',
    bit_depth: Literal[16, 24, 32] = 16
) -> Path:
    """
    High-level function: Load MEG file, convert to audio, save WAV.

    Parameters
    ----------
    input_path : str or Path
        Path to MEG data file (.fif or .mat).
    output_path : str or Path
        Path for output WAV file.
    output_sfreq : int
        Output sampling rate. Default 44100 Hz.
    target_amplitude : float
        Peak amplitude. Default DEFAULT_TARGET_AMPLITUDE.
    channel_indices : list of int, optional
        Indices of channels to include. If None, uses all channels.
    channel_mix : {'mean', 'first', 'all'}
        How to combine channels. Default 'mean'.
    bit_depth : {16, 24, 32}
        Output bit depth. Default 16.

    Returns
    -------
    Path
        Path to saved WAV file.

    Example
    -------
    >>> audify_meg_file(
    ...     "data/raw/subject01.mat",
    ...     "output/subject01.wav",
    ...     channel_indices=[0, 1, 2, 3, 4]  # Occipital channels
    ... )
    """
    from ..preprocessing import load_meg_data

    # Load data
    meg = load_meg_data(input_path)
    data = meg['data']
    sfreq = meg['sfreq']

    if sfreq is None:
        raise ValueError(
            "Sampling frequency not found in file. "
            "Please specify sfreq manually or check file format."
        )

    # Select channels
    if channel_indices is not None:
        data = data[channel_indices, :]

    # Convert to audio
    audio, audio_sfreq = meg_to_audio(
        data,
        sfreq,
        output_sfreq=output_sfreq,
        target_amplitude=target_amplitude,
        channel_mix=channel_mix
    )

    # Save
    return save_audio(audio, audio_sfreq, output_path, bit_depth=bit_depth)
