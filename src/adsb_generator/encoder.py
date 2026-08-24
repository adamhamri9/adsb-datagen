import random
import numpy as np
from enum import Enum

class TXParams(Enum):
    AMPLITUDE = "amplitude"

class ADSBEncoder:
    """
    Converts ADS-B raw frames into baseband I/Q samples with randomized transmission parameters.

    This class handles the encoding of 112-bit ADS-B messages into a complex baseband 
    signal representation, following the pulse-position modulation (PPM) encoding 
    scheme defined in the Mode S/ADS-B standard. The encoder adds configurable 
    random variations to transmission parameters (e.g., amplitude) to simulate 
    real-world signal variability.

    Attributes:
        sample_rate (float): Sampling rate in samples per second.
        tx_params_distributions (dict[TXParams, list[list[float]]]): Mapping 
            of transmission parameters to their probability distributions defined 
            as intervals with associated weights.
        seed (int | None): Seed for the internal random number generator to ensure 
            reproducible signal generation.
    """
    def __init__(self, sample_rate: float = 2e6, tx_params_distributions: dict[TXParams, list[list[float]]] | None = None, seed: int | None = None):
        """
        Initializes the encoder with sampling parameters and transmission parameter distributions.

        Args:
            sample_rate: Sampling rate in samples per second. Defaults to 2 MHz.
            tx_params_distributions: A mapping of TXParams to probability distributions 
                defined as lists of [min_val, max_val, weight] intervals. If None, 
                defaults to a single amplitude distribution with three intervals 
                covering the range [0.05, 1.00].
            seed: Seed for the internal random number generator to ensure reproducible 
                signal generation. If None, the generator is seeded from system randomness.

        Raises:
            ValueError: If `tx_params_distributions` contains invalid keys, invalid 
                intervals (min > max), or weights that do not sum to 1.0 (validated 
                via `_validate_distributions`).
        """
        self.sample_rate = sample_rate

        default_dists = {
                TXParams.AMPLITUDE: [
                    [0.05, 0.25, 0.5],
                    [0.25, 0.65, 0.3],
                    [0.65, 1.00, 0.2]
                ]}

        self.tx_params_dists = tx_params_distributions or default_dists

        self._validate_distributions()

        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._rng = random.Random(self._seed)

    @property
    def seed(self) -> int:
        """Gets the seed value."""
        return self._seed

    def configure(self, sample_rate: float | None = None, tx_params_distributions: dict[TXParams, list[list[float]]] | None = None, seed: int | None = None) -> None:
        """Update sample_rate and/or tx params distributions, random seed, then validate."""
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if tx_params_distributions is not None:
            self.tx_params_dists.update(tx_params_distributions)

        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)

        self._validate_distributions()
    

    def _validate_distributions(self):
        for tx_param, intervals in self.tx_params_dists.items():
            if not isinstance(tx_param, TXParams):
                raise ValueError(
                    f"Invalid tx param key: '{tx_param}'. "
                    "Valid options are the TXParams enums"
                )

            total_weights = 0.0
            for min_val, max_val, weight in intervals:
                if min_val > max_val:
                    raise ValueError(
                        f"Invalid range {min_val} > {max_val} in key '{tx_param}'"
                    )
                total_weights += weight

            if not (0.99 <= total_weights <= 1.01):
                raise ValueError(
                    f"Sum of weights for key '{tx_param}' must equal 1.0, got {total_weights:.2f}"
                )

    def _sample_tx_params(self) -> dict[TXParams, float]:
        sampled_params = {}
        for param_key, intervals in self.tx_params_dists.items(): 
            ranges = [(low, high) for low, high, _ in intervals]
            weights = [w for _, _, w in intervals]

            selected_range = self._rng.choices(ranges, weights, k=1)[0]

            sampled_val = self._rng.uniform(selected_range[0], selected_range[1])

            sampled_params[param_key] = sampled_val

        return sampled_params

    def encode(self, msg: int) -> tuple[np.ndarray, dict[TXParams, float]]:
        """
        Encodes a 112-bit ADS-B message into a complex baseband I/Q signal.

        This method samples transmission parameters from configured distributions and 
        generates a 120 μs signal containing the preamble and 112 data bits encoded 
        using PPM at 1 Mbps.

        The timing follows the ADS-B standard:
            - Preamble: 8.0 μs with pulses at specific positions
            - Data bits: 112 bits at 1 μs per bit starting at 8.0 μs
            - Pulse width: 0.5 μs for both preamble and data pulses

        Args:
            msg: The 112-bit ADS-B message as an integer (LSB alignment).

        Returns:
            A tuple containing:
                - np.ndarray: Complex I/Q samples of the baseband signal (dtype=np.complex64)
                - dict[TXParams, float]: The transmission parameters used for this encode operation
        """
        params = self._sample_tx_params()
        amplitude = params[TXParams.AMPLITUDE]

        samples_per_us = self.sample_rate / 1e6
        total_samples = int(round(120.0 * samples_per_us))

        signal = np.zeros(total_samples, dtype=np.float32)

        for start_us, end_us in ((0.0, 0.5), (1.0, 1.5), (3.5, 4.0), (4.5, 5.0)):
            signal[
                int(round(start_us * samples_per_us)):
                int(round(end_us * samples_per_us))
            ] = amplitude

        bit_start_us = 8.0

        for shift in range(111, -1, -1):
            bit = (msg >> shift) & 1

            pulse_offset_us = 0.0 if bit else 0.5

            p_start = round((bit_start_us + pulse_offset_us) * samples_per_us)
            p_end = round((bit_start_us + pulse_offset_us + 0.5) * samples_per_us)

            signal[p_start:p_end] = amplitude

            bit_start_us += 1.0

        iq_samples = signal.astype(np.complex64)

        return iq_samples, params