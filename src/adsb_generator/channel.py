import random
import numpy as np
from enum import Enum

class ChannelParams(Enum):
    """Supported channel impairment parameters for ADS-B signal simulation."""
    SNR_DB = "snr_db"
    NOISE_CORRELATION = "noise_correlation"

    FREQUENCY_OFFSET = "frequency_offset"
    PHASE_OFFSET = "phase_offset"
    DC_OFFSET_I = "dc_offset_i"
    DC_OFFSET_Q = "dc_offset_q"

    IQ_GAIN_IMBALANCE = "iq_gain_imbalance"
    IQ_PHASE_IMBALANCE = "iq_phase_imbalance"

class ADSBChannel:
    """
    Simulates realistic RF channel impairments on ADS-B baseband I/Q signals.

    This class applies a series of channel effects—including IQ imbalance, DC offset,
    frequency offset, phase offset, and additive white Gaussian noise (AWGN)—to
    simulate the degradations encountered in real-world ADS-B reception. Each
    impairment parameter is sampled from configurable probability distributions.

    Attributes:
        sample_rate (float): Sampling rate in samples per second.
        channel_params_distributions (dict[ChannelParams, list[list[float]]]): 
            Mapping of channel parameters to their probability distributions defined
            as intervals with associated weights.
        seed (int | None): Seed for the internal random number generator to ensure 
            reproducible channel simulation.
    """
    def __init__(self, sample_rate: float = 2e6, channel_params_distributions: dict[ChannelParams, list[list[float]]] | None = None, seed: int | None = None):
        """
        Initializes the channel simulator with sampling parameters and impairment distributions.

        Args:
            sample_rate: Sampling rate in samples per second. Defaults to 2 MHz.
            channel_params_distributions: A mapping of ChannelParams to probability 
                distributions defined as lists of [min_val, max_val, weight] intervals. 
                If None, defaults to distributions covering typical ADS-B reception 
                conditions for all eight channel parameters.
            seed: Seed for the internal random number generator to ensure reproducible 
                channel simulation. If None, the generator is seeded from system randomness.

        Raises:
            ValueError: If `channel_params_distributions` contains invalid keys, invalid 
                intervals (min > max), weights that do not sum to 1.0, or noise correlation 
                values outside [-1.0, 1.0] (validated via `_validate_distributions`).
        """

        self.sample_rate = sample_rate

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

        self._validate_distributions()

        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._rng = random.Random(self._seed)
        self._np_rng = np.random.default_rng(self._seed)
        
    @property
    def seed(self) -> int:
        """Gets the seed value."""
        return self._seed

    def configure(self, sample_rate: float | None = None, channel_params_distributions: dict[ChannelParams, list[list[float]]] | None = None, seed: int | None = None) -> None:
        """Update sample_rate and/or tx params distributions, random seed, then validate."""
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if channel_params_distributions is not None:
            self.channel_params_dists.update(channel_params_distributions)

        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        self._validate_distributions()

    def _validate_distributions(self):
        for channel_param, intervals in self.channel_params_dists.items():
            if not isinstance(channel_param, ChannelParams):
                raise ValueError(
                    f"Invalid channel param key: '{channel_param}'. "
                    "Valid options are the ChannelParams enums"
                )

            total_weights = 0.0
            for min_val, max_val, weight in intervals:
                if min_val > max_val:
                    raise ValueError(
                        f"Invalid range {min_val} > {max_val} in key '{channel_param}'"
                    )

                if channel_param == ChannelParams.NOISE_CORRELATION:
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

        default_values = {
            ChannelParams.SNR_DB: 15.0,
            ChannelParams.NOISE_CORRELATION: 0.0,
            ChannelParams.FREQUENCY_OFFSET: 0.0,
            ChannelParams.PHASE_OFFSET: 0.0,
            ChannelParams.DC_OFFSET_I: 0.0,
            ChannelParams.DC_OFFSET_Q: 0.0,
            ChannelParams.IQ_GAIN_IMBALANCE: 0.0,
            ChannelParams.IQ_PHASE_IMBALANCE: 0.0,
        }

        for param_key, intervals in self.channel_params_dists.items(): 
            ranges = [(low, high) for low, high, _ in intervals]
            weights = [w for _, _, w in intervals]

            selected_range = self._rng.choices(ranges, weights, k=1)[0]

            sampled_val = self._rng.uniform(selected_range[0], selected_range[1])

            sampled_params[param_key] = sampled_val

        for param in ChannelParams:
            if param not in sampled_params:
                sampled_params[param] = default_values[param]

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
        if len(signal) == 0:
            return signal.copy()
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

    def apply(self, signal: np.ndarray) -> tuple[np.ndarray, dict[ChannelParams, float]]:
        """
        Applies simulated channel impairments to an ADS-B baseband I/Q signal.

        This method samples channel parameters from configured distributions and
        applies impairments in the following order: IQ imbalance, DC offset,
        frequency offset, phase offset, and finally additive Gaussian noise.

        Args:
            signal: Complex I/Q samples of the baseband signal (dtype=np.complex64).

        Returns:
            A tuple containing:
                - np.ndarray: The impaired complex I/Q signal (dtype=np.complex64)
                - dict[ChannelParams, float]: The channel parameters used for this operation
        """
        params = self._sample_channel_params()

        signal = self._apply_iq_imbalance(signal, params[ChannelParams.IQ_GAIN_IMBALANCE], params[ChannelParams.IQ_PHASE_IMBALANCE])
        signal = self._apply_dc_offset(signal, params[ChannelParams.DC_OFFSET_I], params[ChannelParams.DC_OFFSET_Q])
        signal = self._apply_freq_offset(signal, params[ChannelParams.FREQUENCY_OFFSET], self.sample_rate)
        signal = self._apply_phase_offset(signal, params[ChannelParams.PHASE_OFFSET])
        signal = self._apply_gaussian_noise(signal, params[ChannelParams.SNR_DB], params[ChannelParams.NOISE_CORRELATION])

        return signal, params

