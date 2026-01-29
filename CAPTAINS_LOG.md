# Captain's Log: MEG Audification Platform

## 2026-01-28: Streaming Architecture Decisions

### The Core Problem
We have 306 MEG channels, 133 seconds of data at 1000 Hz. Users need to:
1. Select a brain region (~38 channels)
2. See streaming waveforms for all channels immediately
3. Hear audified data simultaneously
4. Scrub/seek to any time point (e.g., the eyes-closed transition at 74 sec)

### Data Size Reality Check

| Scope | Samples | Size (Float32) |
|-------|---------|----------------|
| 1 channel, full recording | 133,000 | 532 KB |
| 1 region (38 ch), full | 133,000 × 38 | ~20 MB |
| All 306 MEG channels | 133,000 × 306 | ~163 MB |

### Decision 1: Time-Interleaved Binary Files

**Rejected:** Channel-sequential layout (typical MEG format)
```
[all samples ch0][all samples ch1]...[all samples ch37]
```
Problem: Can't fetch a time window without downloading entire file.

**Chosen:** Time-interleaved layout
```
[ch0_t0, ch1_t0, ...ch37_t0][ch0_t1, ch1_t1, ...ch37_t1]...
```
Benefit: Any time range = contiguous byte range.

### Decision 2: HTTP Range Requests (Not Pre-Chunked Files)

**Rejected:** Pre-chunked files
```
/occipital/chunk_000.bin  (1 sec)
/occipital/chunk_001.bin  (1 sec)
... (133 files per region)
```
Problems: Many files to manage, cache-bust, deploy. Chunk boundary logic.

**Chosen:** Single file per region + Range requests
```
/regions/occipital.bin  (20 MB, one file)

fetch(url, { headers: { 'Range': 'bytes=11248000-11400000' } })
// ^ Fetches exactly second 74, nothing else
```
Benefits:
- Seeking is free (any byte offset)
- Server handles it natively (nginx, S3, CloudFlare)
- No preprocessing into chunks
- One file per region = simple deployment

### Decision 3: Progressive Streaming for Visualization

Don't wait for full file. Use streaming fetch:
```javascript
const reader = response.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  renderPartialData(value);  // Draw waveforms as data arrives
}
```

First 2 seconds (~320 KB) loads in ~20ms → instant waveform render.

### Decision 4: No Resampling for Audification

**Rejected:** scipy.signal.resample to interpolate samples

**Chosen:** Just change the playback sample rate
- Original: 1000 Hz MEG data
- Playback: 44100 Hz audio rate
- Result: Frequencies shift up 44.1×
- 10 Hz alpha → 441 Hz (audible!)
- 133 seconds of data → 3 seconds of audio

This is the same approach used in volcano-audio and space-weather-audio projects.

### File Layout for Deployment

```
/data/
  /regions/
    occipital.bin      # 38 channels × 133000 samples × 4 bytes
    temporal_left.bin
    temporal_right.bin
    frontal.bin
    parietal.bin
    ...
  /metadata/
    regions.json       # Channel indices per region, normalization ranges
```

### Byte Offset Math

For a region with N channels:
```
bytes_per_sample = N × 4  (Float32)
samples_per_second = 1000

To fetch time range [start_sec, end_sec]:
  start_byte = start_sec × 1000 × N × 4
  end_byte = end_sec × 1000 × N × 4
```

Example (38-channel occipital region, second 74-76):
```
start_byte = 74 × 1000 × 38 × 4 = 11,248,000
end_byte = 76 × 1000 × 38 × 4 = 11,552,000
Range: bytes=11248000-11552000  (304 KB for 2 seconds)
```

### Browser Considerations

- HTTP/1.1: 6 concurrent connections per domain (irrelevant with Range requests)
- HTTP/2: Multiplexed, handles multiple ranges efficiently
- Memory: 20 MB per region is fine for modern browsers
- Rendering: 38 waveforms at 60fps is the real bottleneck, not network

### What We're NOT Doing (Yet)

- Pre-chunked files (premature optimization for 133-second demo)
- Compression (zstd would help at scale, not needed now)
- Web Workers for decompression (no compression = no need)
- Progressive chunk sizing like volcano-audio (overkill for demo)

These can be added later when scaling to hours of multi-subject data.

---

## Reference Architecture: volcano-audio

Examined for patterns (in `/reference/volcano-audio/`):
- `js/data-fetcher.js`: Progressive chunk batching algorithm
- `workers/audio-worklet.js`: Real-time audio processing with circular buffer

Key volcano-audio features we might adopt later:
- AudioWorklet for gapless playback with variable speed
- Sample-accurate crossfades for seamless looping
- High-pass filter for DC drift removal
- Anti-aliasing filter for slow playback

For now: simpler Web Audio API approach is sufficient.

---

## 2026-01-28: Real-Time Filtering Decision

### The Insight
The MATLAB file includes both raw data (`data`) and pre-filtered alpha band (`data_filt`).
But pre-filtering locks users into one view of the data.

**Better approach:** Store raw data, filter client-side in real-time.

### Why This Matters
- User moved during recording → movement artifacts in data
- With real-time filtering, users can dial in a high-pass filter and *hear* the artifacts disappear
- Same filter applies to both visualization AND audio simultaneously
- Enables exploration: "What does this data sound like in the theta band? Beta?"

### Architecture

```
Raw Data (from server, unfiltered)
    ↓
┌─────────────────────────────────┐
│  Filter Controls (UI)           │
│  • High-pass: [0.1] - [10] Hz   │
│  • Low-pass:  [30] - [100] Hz   │
│  • Band presets: α β θ δ γ      │
│  • Notch: 60 Hz (power line)    │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Web Worker (filtering)         │
│  - IIR biquad filters           │
│  - Processes all channels       │
│  - Same filtered data → both:   │
└─────────────────────────────────┘
    ↓                    ↓
Waveform Canvas     AudioWorklet
(see the filter)    (hear the filter)
```

### Standard EEG/MEG Frequency Bands

| Band | Frequency | Associated State |
|------|-----------|------------------|
| Delta (δ) | 0.5-4 Hz | Deep sleep |
| Theta (θ) | 4-8 Hz | Drowsiness, memory |
| **Alpha (α)** | **8-13 Hz** | **Relaxation, eyes closed** |
| Beta (β) | 13-30 Hz | Active thinking, focus |
| Gamma (γ) | 30-100 Hz | Perception, cognition |

### Filter Implementation Notes
- Use IIR biquad filters (efficient, low latency)
- Butterworth response for flat passband
- 2nd or 4th order typically sufficient
- volcano-audio already has biquad code in AudioWorklet we can reference

### Validation from Current Data
Eyes closed alpha effect confirmed in MEG2112:
- Eyes OPEN: std = 2.00e-12
- Eyes CLOSED: std = 8.30e-12
- **4.1× stronger alpha with eyes closed**

This should be dramatically audible with alpha band-pass filter applied.

---

## Next Steps

1. Write Python script to export MATLAB data → time-interleaved binaries per region
2. Create region metadata JSON (channel indices, names, normalization ranges)
3. Build minimal web viewer: region selector → waveform display → audio playback
4. Add real-time filter controls with preset bands
5. Test Range request performance on target deployment (university network, CDN)
