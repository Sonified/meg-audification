#!/usr/bin/env python3
"""
Audify MEG data from CSV file.

Usage:
    python scripts/audify_csv.py data/raw/RA_MEG2112.csv
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_meg_data
from src.audification import meg_to_audio, save_audio

# Audio playback sample rate
OUTPUT_SFREQ = 44100


def audify_csv(input_path: str, output_dir: str = None):
    """
    Audify a CSV MEG file.

    Creates one WAV file per data channel.
    """
    input_path = Path(input_path)

    # Default output directory: output/<input_stem>/
    if output_dir is None:
        output_dir = Path("output") / input_path.stem
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {input_path}...")
    meg = load_meg_data(input_path)

    n_channels = meg['data'].shape[0]
    n_samples = meg['data'].shape[1]
    orig_sfreq = meg['sfreq']

    print(f"  Original sample rate: {orig_sfreq:.0f} Hz")
    print(f"  Channels: {n_channels}")
    print(f"  Samples: {n_samples}")
    print(f"  Original duration: {n_samples / orig_sfreq:.1f} seconds")
    print(f"  Audio duration at {OUTPUT_SFREQ} Hz: {n_samples / OUTPUT_SFREQ:.2f} seconds")
    print(f"  Frequency shift: {OUTPUT_SFREQ / orig_sfreq:.1f}x")

    # Audify each channel
    for i in range(n_channels):
        ch_name = meg['ch_names'][i]
        channel_data = meg['data'][i:i+1, :]  # Keep 2D

        audio, sfreq = meg_to_audio(
            channel_data,
            orig_sfreq,
            output_sfreq=OUTPUT_SFREQ,
            highpass_freq=None
        )

        output_path = output_dir / f"{input_path.stem}_{ch_name}.wav"
        save_audio(audio, OUTPUT_SFREQ, output_path)
        print(f"  Saved: {output_path}")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    audify_csv(input_file, output_dir)
