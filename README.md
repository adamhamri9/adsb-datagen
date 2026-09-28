# adsb-generator

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10-3.13](https://img.shields.io/badge/Python-3.10--3.13-3776AB.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/adamhamri9/adsb-datagen)

**Synthetic ADS-B baseband signal generator with configurable RF channel simulation.**

`adsb-generator` produces realistic I/Q (In-phase/Quadrature) samples of ADS-B (Mode S Downlink Format 17) baseband signals, along with the raw 112-bit message and all applied parameters. Each sample carries three signal stages: the ideal PPM waveform, the waveform after transmitter impairments, and the waveform after RF channel impairments. It implements an infinite iterator that streams reproducible samples with full control over message type distributions, transmission parameters, and channel impairments.

## Key Features

- **End-to-end pipeline**: random message generation, PPM encoding, and RF channel simulation in a single call.
- **Four ADS-B message types**: identification, surface position, airborne position, and airborne velocity with configurable emission probabilities.
- **Realistic channel impairments**: Gaussian noise (both AWGN and correlated), frequency/phase offset, IQ imbalance, and DC offset, all sampled from configurable probability distributions.
- **Transmitter impairments**: linear amplitude droop and filtered phase noise applied between encoding and the channel stage.
- **Reproducibility**: deterministic output via a shared seed across all pipeline stages.
- **Configurable distributions**: override any transmission or channel parameter distribution to model specific receiver conditions or hardware behavior.
- **NumPy-native**: signals are NumPy arrays, ready for direct use with any downstream processing tool. The ideal PPM signal is `np.complex64`; note that the amplitude droop stage promotes later stages to `complex128`.
- **Export to multiple formats**: save samples as `.npz` (full signal data), `.csv`, `.json`, or `.jsonl` (metadata only).

## Requirements

- Python 3.10 -- 3.13
- NumPy >= 1.21.3

## Installation

```bash
pip install adsb-generator
```

## Usage

```python
from adsb_generator import ADSBGenerator

# Create a generator with a fixed seed for reproducibility
gen = ADSBGenerator(seed=42)

# Each iteration yields an ADSBSample carrying the message and three
# signal stages: clean (ideal PPM), transmitted (after TX impairments),
# and channel (after RF impairments)
for sample in gen:
    print(f"Message type   : {sample.message_type.value}")
    print(f"Raw message    : {sample.message:#028x}")
    print(f"Clean signal   : {sample.clean_signal.shape} {sample.clean_signal.dtype}")
    print(f"Transmitted    : {sample.transmitted_signal.shape} {sample.transmitted_signal.dtype}")
    print(f"Channel signal : {sample.channel_signal.shape} {sample.channel_signal.dtype}")
    print(f"SNR (dB)       : {sample.channel_params['snr_db']:.1f}")
    print(f"Amplitude      : {sample.tx_params['amplitude']:.3f}")
    print(f"Amplitude droop: {sample.tx_params['amplitude_droop']:.3f}")
    break
```

### Customizing Distributions

```python
from adsb_generator import ADSBGenerator, MessageType, ChannelParams

# Favor airborne positions, restrict SNR to low-moderate range
gen = ADSBGenerator(
    message_type_probs={
        MessageType.AIRBORNE_POSITION: 0.60,
        MessageType.AIRBORNE_VELOCITY: 0.20,
        MessageType.IDENTIFICATION: 0.10,
        MessageType.SURFACE_POSITION: 0.10,
    },
    channel_params_distributions={
        ChannelParams.SNR_DB: [
            [3.0, 8.0, 0.70],
            [8.0, 15.0, 0.30],
        ],
    },
    sample_rate=2e6,
    seed=12345,
)

sample = next(gen)
```

### Updating Configuration at Runtime

Use `configure()` to update distributions, sample rate, or seed on an already-created generator without reconstructing it.

```python
from adsb_generator import ADSBGenerator, MessageType, ChannelParams, TXParams

gen = ADSBGenerator(seed=42)

# Generate some samples with default configuration
sample = next(gen)

# Switch to only airborne messages with high SNR
gen.configure(
    message_type_probs={
        MessageType.AIRBORNE_POSITION: 1.0,
    },
    channel_params_distributions={
        ChannelParams.SNR_DB: [[20.0, 25.0, 1.0]],
    },
    seed=99,
)

# Next samples follow the new configuration
sample = next(gen)
assert sample.message_type == MessageType.AIRBORNE_POSITION
```

You can also configure individual components directly:

```python
from adsb_generator import ADSBGenerator

gen = ADSBGenerator(seed=42)

# ADSBTransmitter
gen.transmitter.configure(sample_rate=4e6)

# ADSBChannel
gen.channel.configure(
    channel_params_distributions={
        ChannelParams.SNR_DB: [[10.0, 15.0, 1.0]],
        ChannelParams.FREQUENCY_OFFSET: [[0.0, 0.0, 1.0]],
    },
    seed=99,

# ADSBMessage
gen.builder.configure(seed=55)
)
```

### Handling Missing Parameters

When you only specify a subset of parameters, `fill_missing()` controls how the rest are handled.

```python
from adsb_generator import ADSBGenerator, MissingPolicy, ChannelParams, TXParams

# Provide only SNR -- other channel params are missing
partial_dists = {ChannelParams.SNR_DB: [[10.0, 15.0, 1.0]]}
gen = ADSBGenerator(channel_params_distributions=partial_dists, seed=42)

# Option 1: IGNORE -- missing params default to 0.0
gen.fill_missing(MissingPolicy.IGNORE)
sample = next(gen)

# Option 2: DEFAULTS -- fill missing params from built-in defaults
gen.fill_missing(MissingPolicy.DEFAULTS)

# Option 3: CONSTANTS -- provide exact values for missing params
gen.fill_missing(MissingPolicy.CONSTANTS, channel_values={
    ChannelParams.FREQUENCY_OFFSET: 500.0,
    ChannelParams.PHASE_OFFSET: 0.0,
    ChannelParams.DC_OFFSET_I: 0.0,
    ChannelParams.DC_OFFSET_Q: 0.0,
    ChannelParams.IQ_GAIN_IMBALANCE: 0.01,
    ChannelParams.IQ_PHASE_IMBALANCE: 0.5,
    ChannelParams.NOISE_CORRELATION: 0.0,
})

# Option 4: RAISE -- error if any are missing (strict mode)
gen.fill_missing(MissingPolicy.RAISE)
```

The same applies to transmission parameters:

```python
from adsb_generator import ADSBTransmitter, MissingPolicy, TXParams

transmitter = ADSBTransmitter(seed=42)
transmitter.fill_missing(MissingPolicy.CONSTANTS, values={
    TXParams.AMPLITUDE: 0.8,
    TXParams.AMPLITUDE_DROOP: 0.05,
    TXParams.PHASE_NOISE_LEVEL: 0.0,
    TXParams.PHASE_NOISE_BANDWIDTH: 1e4,
})
```

### Exporting Samples

Use `export()` to save generated samples to disk. Pass samples directly or use buffering to accumulate them first.

```python
from adsb_generator import ADSBGenerator

gen = ADSBGenerator(seed=42)

# Export directly by passing samples
samples = gen.generate(100)
gen.export("data.npz", samples=samples)

# Or use the buffer to accumulate samples across multiple generate calls
gen.start_buffering()
gen.generate(50)
gen.generate(50)
gen.export("data.csv")
```

Four output formats are supported:

| Format | Extension | Signal data | Description |
|---|---|---|---|
| NumPy | `.npz` | Included | Full archive with all sample data including all three I/Q signal stages. |
| CSV | `.csv` | Excluded | Flat table with metadata and flattened parameter columns. |
| JSON | `.json` | Excluded | Array of sample objects with compact formatting. |
| JSONL | `.jsonl` | Excluded | One JSON object per line, suitable for streaming ingestion. |

Only the `.npz` format preserves `clean_signal`, `transmitted_signal`, and `channel_signal`. The other three formats export metadata only: the message, message type, and the flattened transmission and channel parameters.

```python
# Save full signal data for offline processing
gen.export("samples.npz", samples=samples)

# Save metadata-only CSV for analysis
gen.export("samples.csv", samples=samples)

# Save as JSON for web APIs or inspection
gen.export("samples.json", samples=samples)

# Save as JSONL for streaming pipelines
gen.export("samples.jsonl", samples=samples)
```

By default, `export()` clears the internal buffer after writing. Pass `clear=False` to keep the buffer intact:

```python
gen.start_buffering()
gen.generate(100)
gen.export("data.npz")        # clears buffer
gen.export("data.csv")        # buffer is empty, raise ValueError
```

## API Reference

### `ADSBGenerator`

```python
ADSBGenerator(
    message_type_probs: dict[MessageType | str, float] | None = None,
    tx_params_distributions: dict[TXParams | str, list[list[float]]] | None = None,
    channel_params_distributions: dict[ChannelParams | str, list[list[float]]] | None = None,
    sample_rate: float = 2e6,
    seed: int | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `message_type_probs` | `dict` or `None` | Equal 25% per type | Mapping of `MessageType` to emission probabilities (must sum to 1.0). |
| `tx_params_distributions` | `dict` or `None` | Built-in amplitude, droop, and phase noise bands | Mapping of `TXParams` to `[[min, max, weight], ...]` intervals. |
| `channel_params_distributions` | `dict` or `None` | Typical ADS-B conditions | Mapping of `ChannelParams` to `[[min, max, weight], ...]` intervals. |
| `sample_rate` | `float` | `2e6` | Sampling rate in samples per second. |
| `seed` | `int` or `None` | Random | Seed for deterministic output across all pipeline stages. |

| Method | Returns | Description |
|---|---|---|
| `generate(n=1)` | `list[ADSBSample]` or `None` | Generates `n` samples. Returns `None` when buffering is enabled. |
| `export(path, samples=None, clear=True)` | `None` | Saves samples to file. Supported: `.npz`, `.csv`, `.json`, `.jsonl`. |
| `start_buffering()` | `None` | Enables buffering mode so `generate()` appends to an internal buffer. |
| `stop_buffering(clear=True)` | `None` | Disables buffering. If `clear=True`, clears the buffer. |
| `configure(...)` | `None` | Updates message, channel, and encoder configurations. |
| `fill_missing(policy, ...)` | `None` | Sets the missing parameter handling policy. |
| `reset()` | `None` | Resets sample rate and all sub-components to initial values. |
| `clone(seed=None)` | `ADSBGenerator` | Creates a new generator with the same configuration. |

---

### `ADSBSample`

Dataclass returned by each iteration of `ADSBGenerator`.

| Field | Type | Description |
|---|---|---|
| `message` | `int` | Complete 112-bit ADS-B message (including 24-bit CRC) as an integer. |
| `message_type` | `MessageType` | The type of ADS-B message generated. |
| `clean_signal` | `np.ndarray` | Complex baseband I/Q signal as ideal PPM, before any transmission impairment (`complex64`). |
| `transmitted_signal` | `np.ndarray` | Complex baseband I/Q signal after transmission impairments: amplitude droop, then phase noise. |
| `tx_params` | `dict[TXParams, float]` | Transmission parameters applied during transmission. |
| `channel_signal` | `np.ndarray` | Complex baseband I/Q signal after channel impairments are applied to `transmitted_signal`. |
| `channel_params` | `dict[ChannelParams, float]` | Channel parameters applied to the signal. |

The three signal stages form a chain: `clean_signal` -> `transmitted_signal` -> `channel_signal`. Each stage is the input to the next, so channel impairments operate on the already-drooped and phase-noised waveform.

---

### `MessageType`

| Value | Description |
|---|---|
| `IDENTIFICATION` | Aircraft identification (callsign) messages (TC 1--4). |
| `SURFACE_POSITION` | Surface position messages (TC 5--8). |
| `AIRBORNE_POSITION` | Airborne position messages (TC 9--18, 20--22). |
| `AIRBORNE_VELOCITY` | Airborne velocity messages (TC 19). |

---

### `TXParams`

| Value | Description |
|---|---|
| `AMPLITUDE` | Transmitted signal amplitude. |
| `AMPLITUDE_DROOP` | Fractional linear amplitude decay across the burst, in `[0.0, 1.0]`. A value of `0.1` scales the last sample to 90% of full amplitude. |
| `PHASE_NOISE_LEVEL` | Standard deviation of the applied phase noise, in radians. |
| `PHASE_NOISE_BANDWIDTH` | One-sided bandwidth in Hz of the first-order phase noise filter. Clamped to `0.499 * sample_rate`. |

Impairments are applied in a fixed order: amplitude droop, then phase noise.

---

### `ADSBTransmitter`

Encodes 112-bit messages into complex baseband I/Q samples using PPM per the Mode S standard, then applies transmission impairments.

```python
ADSBTransmitter(
    sample_rate: float = 2e6,
    tx_params_distributions: dict[TXParams | str, list[list[float]]] | None = None,
    seed: int | None = None,
)
```

| Method | Returns | Description |
|---|---|---|
| `transmit(msg: int)` | `tuple[np.ndarray, np.ndarray, dict[TXParams, float]]` | Encodes a 112-bit message into a 120-us baseband I/Q signal and returns the ideal signal, the impaired signal, and the parameters used. |

`transmit()` returns:

| Element | Type | Description |
|---|---|---|
| `[0]` | `np.ndarray` | Ideal PPM signal before transmission impairments. |
| `[1]` | `np.ndarray` | Signal after amplitude droop and phase noise. |
| `[2]` | `dict[TXParams, float]` | Transmission parameters used for this call. |

```python
from adsb_generator import ADSBTransmitter

tx = ADSBTransmitter(seed=42)

clean, transmitted, params = tx.transmit(0xDEADBEEF)
```

---

### `ADSBChannel`

Simulates realistic RF channel impairments on baseband I/Q signals.

```python
ADSBChannel(
    sample_rate: float = 2e6,
    channel_params_distributions: dict[ChannelParams | str, list[list[float]]] | None = None,
    seed: int | None = None,
)
```

| Method | Returns | Description |
|---|---|---|
| `apply(signal: np.ndarray)` | `tuple[np.ndarray, dict[ChannelParams, float]]` | Applies impairments: IQ imbalance, DC offset, frequency offset, phase offset, AWGN. |

---

### `ChannelParams`

| Value | Description |
|---|---|
| `SNR_DB` | Signal-to-noise ratio in dB. |
| `NOISE_CORRELATION` | I/Q noise correlation coefficient (-1.0 to 1.0). |
| `FREQUENCY_OFFSET` | Carrier frequency offset in Hz. |
| `PHASE_OFFSET` | Phase offset in radians. |
| `DC_OFFSET_I` | DC offset on the in-phase component. |
| `DC_OFFSET_Q` | DC offset on the quadrature component. |
| `IQ_GAIN_IMBALANCE` | Gain imbalance between I and Q channels. |
| `IQ_PHASE_IMBALANCE` | Phase imbalance between I and Q channels (degrees). |

---

### `ADSBAlgorithms`

Static utility class providing ADS-B encoding algorithms.

| Method | Returns | Description |
|---|---|---|
| `calculate_crc(data: int)` | `int` | Computes 24-bit CRC parity using polynomial `0xFFF409`. |
| `encode_cpr(lat, lon, odd)` | `tuple[int, int]` | Encodes lat/lon into CPR 17-bit values. |
| `encode_altitude(alt: int)` | `int` | Encodes altitude in feet into 12-bit Gillham-coded format. |
| `encode_ground_track(degrees, valid)` | `tuple[int, int]` | Encodes ground track heading into the 7-bit format. |

## Distribution Format

All configurable distributions use the format `[[min_val, max_val, weight], ...]` where weights for a given parameter must sum to `1.0` (+/- 0.01 tolerance). Keys can be enum members.

## License

[MIT](LICENSE) -- Copyright (c) 2026 Adam Hamri (adamhamri9)
