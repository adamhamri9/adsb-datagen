# adsb-generator

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/downloads/)

**Synthetic ADS-B data generator.**

`adsb-generator` produces realistic I/Q (In-phase/Quadrature) samples of clean and impaired ADS-B (Mode S Downlink Format 17) baseband signals, along with the raw 112-bit message and all applied parameters. It implements an infinite iterator that streams reproducible samples with full control over message type distributions, transmission parameters, and channel impairments.

## Key Features

- **End-to-end pipeline**: random message generation, PPM encoding, and RF channel simulation in a single call.
- **Four ADS-B message types**: identification, surface position, airborne position, and airborne velocity with configurable emission probabilities.
- **Realistic channel impairments**: Gaussian Noise(both AWGN and Correlated) , frequency & phase offset, IQ imbalance, and DC offset, all sampled from configurable probability distributions.
- **Reproducibility**: deterministic output via a shared seed across all pipeline stages.
- **Configurable distributions**: override any transmission or channel parameter distribution to model specific receiver conditions or hardware behavior.
- **NumPy-native**: all signals are `np.complex64` arrays, ready for direct use with any downstream processing tool.

## Requirements

- Python 3.10+
- NumPy

## Installation

```bash
pip install adsb-generator
```

## Usage

```python
from adsb_generator import ADSBGenerator

# Create a generator with a fixed seed for reproducibility
gen = ADSBGenerator(seed=42)

# Each iteration yields an ADSBSample with the raw message,
# clean signal, and channel-impaired signal
for sample in gen:
    print(f"Message type : {sample.message_type.value}")
    print(f"Raw message  : {sample.message:#028x}")
    print(f"Clean signal : {sample.clean_signal.shape} complex64 samples")
    print(f"Noisy signal : {sample.channel_signal.shape} complex64 samples")
    print(f"SNR (dB)     : {sample.channel_params['snr_db']:.1f}")
    print(f"Amplitude    : {sample.tx_params['amplitude']:.3f}")
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
| `tx_params_distributions` | `dict` or `None` | Single amplitude band | Mapping of `TXParams` to `[[min, max, weight], ...]` intervals. |
| `channel_params_distributions` | `dict` or `None` | Typical ADS-B conditions | Mapping of `ChannelParams` to `[[min, max, weight], ...]` intervals. |
| `sample_rate` | `float` | `2e6` | Sampling rate in samples per second. |
| `seed` | `int` or `None` | Random | Seed for deterministic output across all pipeline stages. |

---

### `ADSBSample`

Dataclass returned by each iteration of `ADSBGenerator`.

| Field | Type | Description |
|---|---|---|
| `message` | `int` | Complete 112-bit ADS-B message (including 24-bit CRC) as an integer. |
| `message_type` | `MessageType` | The type of ADS-B message generated. |
| `clean_signal` | `np.ndarray` | Complex baseband I/Q signal before channel impairments (`complex64`). |
| `tx_params` | `dict[TXParams, float]` | Transmission parameters applied during encoding. |
| `channel_signal` | `np.ndarray` | Complex baseband I/Q signal after channel impairments (`complex64`). |
| `channel_params` | `dict[ChannelParams, float]` | Channel parameters applied to the signal. |

---

### `MessageType`

| Value | Description |
|---|---|
| `IDENTIFICATION` | Aircraft identification (callsign) messages (TC 1--4). |
| `SURFACE_POSITION` | Surface position messages (TC 5--8). |
| `AIRBORNE_POSITION` | Airborne position messages (TC 9--18, 20--22). |
| `AIRBORNE_VELOCITY` | Airborne velocity messages (TC 19). |

---

### `ADSBEncoder`

Encodes 112-bit messages into complex baseband I/Q samples using PPM per the Mode S standard.

```python
ADSBEncoder(
    sample_rate: float = 2e6,
    tx_params_distributions: dict[TXParams | str, list[list[float]]] | None = None,
    seed: int | None = None,
)
```

| Method | Returns | Description |
|---|---|---|
| `encode(msg: int)` | `tuple[np.ndarray, dict[TXParams, float]]` | Encodes a 112-bit message into a 120-us baseband I/Q signal. |

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

All configurable distributions use the format `[[min_val, max_val, weight], ...]` where weights for a given parameter must sum to `1.0` (+/- 0.01 tolerance). Keys can be enum members or their string values (e.g., `ChannelParams.SNR_DB` or `"snr_db"`).

## License

[MIT](LICENSE) - Copyright (c) 2026 Adam Hamri (adamhamri9)
