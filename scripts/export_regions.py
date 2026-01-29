#!/usr/bin/env python3
"""
Export MEG data to time-interleaved binary files for web streaming.

Each region file contains all channels for that region, organized by time:
  [ch0_t0, ch1_t0, ..., chN_t0, ch0_t1, ch1_t1, ..., chN_t1, ...]

This layout enables HTTP Range requests for any time window.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_meg_data

# Region definitions based on Neuromag sensor layout
# Channel numbers are the numeric part of MEGxxxx names
REGIONS = {
    'left_frontal': {
        'description': 'Left Frontal',
        'channel_prefixes': ['031', '032', '033', '051', '052', '053', '054'],
    },
    'right_frontal': {
        'description': 'Right Frontal',
        'channel_prefixes': ['091', '092', '093', '094', '121', '122', '123', '124'],
    },
    'left_temporal': {
        'description': 'Left Temporal',
        'channel_prefixes': ['011', '012', '013', '014', '015', '141', '142', '151', '152', '153', '154', '155'],
    },
    'right_temporal': {
        'description': 'Right Temporal',
        'channel_prefixes': ['141', '142', '143', '144'],
    },
    'left_parietal': {
        'description': 'Left Parietal',
        'channel_prefixes': ['041', '042', '043', '044', '071', '072', '073', '181', '182', '183'],
    },
    'right_parietal': {
        'description': 'Right Parietal',
        'channel_prefixes': ['072', '073', '074', '221', '222', '223', '224'],
    },
    'left_occipital': {
        'description': 'Left Occipital',
        'channel_prefixes': ['171', '172', '173', '191', '192', '193', '194', '211'],
    },
    'right_occipital': {
        'description': 'Right Occipital',
        'channel_prefixes': ['231', '232', '233', '234', '241', '242', '243', '251', '252', '261', '262', '263'],
    },
    'occipital_all': {
        'description': 'All Occipital (Left + Right)',
        'channel_prefixes': ['171', '172', '173', '191', '192', '193', '194', '211', '212', '213',
                            '231', '232', '233', '234', '241', '242', '243', '251', '252', '261', '262', '263'],
    },
}


def get_channel_indices(ch_names, prefixes):
    """Find channel indices matching the given prefixes."""
    indices = []
    matched_names = []

    for i, name in enumerate(ch_names):
        if not name.startswith('MEG'):
            continue
        # Extract numeric part (e.g., 'MEG2112' -> '211')
        num = name[3:6]  # First 3 digits after 'MEG'
        if num in prefixes:
            indices.append(i)
            matched_names.append(name)

    return indices, matched_names


def export_region(data, ch_names, region_name, region_info, output_dir, sfreq):
    """Export a single region to time-interleaved binary."""

    indices, matched_names = get_channel_indices(ch_names, region_info['channel_prefixes'])

    if not indices:
        print(f"  WARNING: No channels found for {region_name}")
        return None

    # Extract region data: (n_channels, n_samples)
    region_data = data[indices, :]
    n_channels, n_samples = region_data.shape

    # Transpose to time-major: (n_samples, n_channels)
    interleaved = region_data.T.astype(np.float32)

    # Save binary
    output_path = output_dir / f"{region_name}.bin"
    interleaved.tofile(output_path)

    file_size = output_path.stat().st_size

    print(f"  {region_name}: {n_channels} channels, {n_samples} samples")
    print(f"    Channels: {matched_names[:5]}{'...' if len(matched_names) > 5 else ''}")
    print(f"    File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"    Saved: {output_path}")

    return {
        'name': region_name,
        'description': region_info['description'],
        'channels': matched_names,
        'channel_indices': indices,
        'n_channels': n_channels,
        'n_samples': n_samples,
        'sample_rate': sfreq,
        'duration_seconds': n_samples / sfreq,
        'bytes_per_sample': n_channels * 4,  # float32
        'file_size_bytes': file_size,
        'file': f"{region_name}.bin",
    }


def main():
    # Paths
    mat_path = Path('data/raw/dropbox_data/RA_MEG_eyes_open_eyes_closed/RA_260127_MEG_eyes_open_eyes_closed.mat')
    output_dir = Path('data/processed/regions')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {mat_path}...")
    meg = load_meg_data(mat_path)

    mat = meg['mat_contents']
    data = mat['data']  # Raw data (308, 133000)
    ch_names = list(mat['raw_struct'].info.ch_names)
    sfreq = 1000.0  # 1000 Hz

    print(f"Data shape: {data.shape}")
    print(f"Sample rate: {sfreq} Hz")
    print(f"Duration: {data.shape[1] / sfreq:.1f} seconds")
    print()

    # Export each region
    metadata = {
        'source_file': str(mat_path),
        'sample_rate': sfreq,
        'total_samples': data.shape[1],
        'total_channels': len(ch_names),
        'duration_seconds': data.shape[1] / sfreq,
        'regions': {}
    }

    print("Exporting regions...")
    for region_name, region_info in REGIONS.items():
        region_meta = export_region(data, ch_names, region_name, region_info, output_dir, sfreq)
        if region_meta:
            metadata['regions'][region_name] = region_meta
        print()

    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")

    # Summary
    print("\n=== SUMMARY ===")
    total_size = sum(r['file_size_bytes'] for r in metadata['regions'].values())
    print(f"Total regions: {len(metadata['regions'])}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"\nTo test locally:")
    print(f"  cd {output_dir}")
    print(f"  python -m http.server 8080")
    print(f"  # Then fetch with Range headers from http://localhost:8080/")


if __name__ == '__main__':
    main()
