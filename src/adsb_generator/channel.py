import random
import numpy as np
from enum import Enum

class ChannelParams(Enum):
    SNR_DB = "snr_db"
    NOISE_CORRELATION = "noise_correlation"

    FREQUENCY_OFFSET = "frequency_offset"
    PHASE_OFFSET = "phase_offset"
    DC_OFFSET_I = "dc_offset_i"
    DC_OFFSET_Q = "dc_offset_q"

    IQ_GAIN_IMBALANCE = "iq_gain_imbalance"
    IQ_PHASE_IMBALANCE = "iq_phase_imbalance"

class ADSBChannel:
    def __init__(self, channel_params_distributions: (dict[ChannelParams | str, list[list[float]]]) | None = None, seed: int | None = None):

        default_dists = {
            ChannelParams.SNR_DB: [
                [3.0, 8.0, 0.20],
                [8.0, 15.0, 0.55],
                [15.0, 25.0, 0.25],
            ],
            ChannelParams.NOISE_CORRELATION: [
                [0.0, 0.0, 1.0],
            ],
            ChannelParams.FREQUENCY_OFFSET: [
                [-1000.0, 1000.0, 0.80],
                [-3000.0, 3000.0, 0.20],
            ],
            ChannelParams.PHASE_OFFSET: [
                [-0.10, 0.10, 0.80],
                [-0.30, 0.30, 0.20],
            ],
            ChannelParams.DC_OFFSET_I: [
                [-0.01, 0.01, 0.90],
                [-0.03, 0.03, 0.10],
            ],
            ChannelParams.DC_OFFSET_Q: [
                [-0.01, 0.01, 0.90],
                [-0.03, 0.03, 0.10],
            ],
            ChannelParams.IQ_GAIN_IMBALANCE: [
                [0.00, 0.02, 0.80],
                [0.02, 0.05, 0.20],
            ],
            ChannelParams.IQ_PHASE_IMBALANCE: [
                [-1.0, 1.0, 0.80],
                [-3.0, 3.0, 0.20],
            ],
        }

        self.channel_params_dists = channel_params_distributions or default_dists

        self._validate_distributions

        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._rng = random.Random(self._seed)
        self._np_rng = np.random.default_rng(self._seed)
        
    @property
    def seed(self) -> int:
        """Gets the seed value passed at initialization."""
        return self._seed

    def _validate_distributions(self):
        valid_types = set(ChannelParams)
        valid_values = {item.value for item in ChannelParams}

        for channel_param, intervals in self.channel_params_dists.items():
            if channel_param not in valid_types and channel_param not in valid_values:
                raise ValueError(
                    f"Invalid channel param key: '{channel_param}'. "
                    f"Valid options are the ChannelParams enums or: {valid_values}"
                )

            enum_key = channel_param if isinstance(channel_param, ChannelParams) else ChannelParams(channel_param)

            total_weights = 0.0
            for min_val, max_val, weight in intervals:
                if min_val > max_val:
                    raise ValueError(
                        f"Invalid range {min_val} > {max_val} in key '{channel_param}'"
                    )

                if enum_key == ChannelParams.NOISE_CORRELATION:
                    if not (-1.0 <= min_val <= 1.0) or not (-1.0 <= max_val <= 1.0):
                        raise ValueError(
                            f"Noise correlation must be in range [-1.0, 1.0]. "
                            f"Got range [{min_val}, {max_val}] for key '{channel_param}'"
                        )
                    if min_val > max_val:
                        raise ValueError(
                            f"Invalid noise correlation range: {min_val} > {max_val}"
                        )
                total_weights += weight

            if not (0.99 <= total_weights <= 1.01):
                raise ValueError(
                    f"Sum of weights for key '{channel_param}' must equal 1.0, got {total_weights:.2f}"
                )

    def _sample_channel_params(self) -> dict[ChannelParams, float]:
        sampled_params = {}
        for param_key, intervals in self.channel_params_dists.items(): 
            ranges = [(low, high) for low, high, _ in intervals]
            weights = [w for _, _, w in intervals]

            selected_range = self._rng.choices(ranges, weights, k=1)[0]

            sampled_val = self._rng.uniform(selected_range[0], selected_range[1])

            enum_key = param_key if isinstance(param_key, ChannelParams) else ChannelParams(param_key)
            sampled_params[enum_key] = sampled_val

        return sampled_params

    def _apply_dc_offset(self, signal: np.ndarray, dc_offset_i: float, dc_offset_q: float) -> np.ndarray:
        dc_offset = dc_offset_i + 1j * dc_offset_q
        return signal + dc_offset
    
    def _apply_freq_offset(self, signal: np.ndarray, freq_offset: float, sample_rate: float) -> np.ndarray:
        t = np.arange(len(signal)) / sample_rate
        return signal * np.exp(1j * 2 * np.pi * freq_offset * t)

    def _apply_phase_offset(self, signal: np.ndarray, phase_offset: float) -> np.ndarray:
        return signal * np.exp(1j * phase_offset)

    def _apply_iq_imbalance(self, signal: np.ndarray, gain_imbalance: float, phase_imbalance: float) -> np.ndarray:
        phi = np.deg2rad(phase_imbalance)
        
        i = signal.real * (1 + gain_imbalance / 2)
        q = signal.imag * (1 - gain_imbalance / 2)
        
        i_out = i * np.cos(phi) + q * np.sin(phi)
        q_out = i * np.sin(phi) + q * np.cos(phi)
        
        return i_out + 1j * q_out

    def _apply_gaussian_noise(self, signal: np.ndarray, snr_db: float, noise_correlation: float = 0) -> np.ndarray:
        signal_power = np.mean(np.abs(signal) ** 2)
        
        snr_linear = 10 ** (snr_db / 10)
        
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        
        if noise_correlation > 0:
            cov_matrix = noise_power / 2 * np.array([
                [1.0, noise_correlation],
                [noise_correlation, 1.0]
            ])
            
            noise = self._np_rng.multivariate_normal(
                mean=[0, 0],
                cov=cov_matrix,
                size=len(signal)
            )
            noise_complex = noise[:, 0] + 1j * noise[:, 1]
        else:
            noise_i = self._np_rng.normal(0, noise_std, len(signal))
            noise_q = self._np_rng.normal(0, noise_std, len(signal))
            noise_complex = noise_i + 1j * noise_q
        
        return signal + noise_complex
