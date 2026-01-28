# MEG Audification

Converting magnetoencephalography (MEG) brain signals to sound, with a focus on occipital lobe activity.

## Overview

This project transforms MEG data into audible sound (audification), enabling researchers to explore brain activity through listening. The primary focus is on the ~20 channels capturing occipital lobe data from a standard 306-channel MEG system.

## Features

- **MEG Data Processing**: Load and preprocess MEG data using MNE-Python
- **Channel Selection**: Extract and focus on occipital lobe channels
- **Audification Pipeline**: Convert neural signals to audio
- **Dual Output**: WAV file export and real-time playback

## Project Structure

```
meg-audification/
├── data/                  # MEG data files (not tracked in git)
│   ├── raw/               # Original MEG recordings
│   └── processed/         # Preprocessed data
├── src/                   # Source code
│   ├── preprocessing/     # MEG data loading and cleaning
│   ├── audification/      # Signal-to-sound conversion
│   └── playback/          # Real-time audio streaming
├── output/                # Generated audio files
├── notebooks/             # Jupyter notebooks for exploration
└── tests/                 # Unit tests
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Sonified/meg-audification.git
cd meg-audification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- **MNE-Python**: MEG/EEG data processing
- **NumPy/SciPy**: Numerical computing and signal processing
- **sounddevice**: Real-time audio playback
- **soundfile**: Audio file I/O

## Usage

```python
from src.preprocessing import load_meg_data
from src.audification import audify_channels

# Load MEG data and extract occipital channels
meg_data = load_meg_data("data/raw/recording.fif")
occipital = meg_data.pick_channels(occipital_channels)

# Convert to audio
audio = audify_channels(occipital)
audio.save("output/occipital_audification.wav")
audio.play()  # Real-time playback
```

## Data

MEG data files are not included in this repository due to size and privacy considerations. Place your `.fif` files in the `data/raw/` directory.

## License

MIT License

## Acknowledgments

Part of the [Sonified](https://github.com/Sonified) project.
