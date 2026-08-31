import random
import numpy as np
from enum import Enum
from .types import MissingPolicy, TXParams

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

        self.default_dists = {
                TXParams.AMPLITUDE: [
                    [0.05, 0.25, 0.5],
                    [0.25, 0.65, 0.3],
                    [0.65, 1.00, 0.2]
                ]}

        self.tx_params_dists = tx_params_distributions or self.default_dists

        self.missing_policy = MissingPolicy.IGNORE
        self.constant_values = {}

        self._validate_distributions()

        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._rng = random.Random(self._seed)

        self._initial_sample_rate = sample_rate
        self._initial_tx_params_distributions = self.tx_params_dists
        self._initial_seed = self._seed

    @property
    def seed(self) -> int:
        """Gets the seed value."""
        return self._seed

    def configure(self, sample_rate: float | None = None, tx_params_distributions: dict[TXParams, list[list[float]]] | None = None, seed: int | None = None, update_initial: bool = False) -> None:
        """Update sample_rate and/or tx params distributions, random seed, then validate."""
        if sample_rate is not None:
            self.sample_rate = sample_rate
            self._initial_sample_rate = sample_rate if update_initial else self._initial_sample_rate

        if tx_params_distributions is not None:
            self.tx_params_dists.update(tx_params_distributions)
            self._initial_tx_params_distributions = tx_params_distributions if update_initial else self._initial_tx_params_distributions

        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)
            self._initial_seed = seed if update_initial else self._initial_seed

        self._validate_distributions()

    def fill_missing(self, policy: MissingPolicy, values: dict[TXParams, float] | None = None) -> None:
        """
        Configure the missing tx parameters handling policy.

        Args:
            policy: The missing policy to apply (RAISE, IGNORE, DEFAULTS, or CONSTANTS).
            values: Required when policy is CONSTANTS. Maps each TXParams key to its constant value.
        """
        self.missing_policy = policy
        missing_keys = self._get_missing_keys()

        if policy == MissingPolicy.DEFAULTS:
            self.tx_params_dists.update(self.default_dists)
        elif policy == MissingPolicy.CONSTANTS:
            if values is None:
                raise ValueError(
                    "Constant values are required when policy is CONSTANTS. "
                    "Please provide a 'values' dictionary mapping each TXParams key to its constant value."
                )
            for key in missing_keys:
                if key in values:
                    self.constant_values[key] = values[key]
                else:
                    raise ValueError(
                        f"Missing constant value for '{key.value}'. "
                        f"Please provide a value in the 'values' dictionary."
                    )
        elif policy == MissingPolicy.RAISE:
            self._validate_distributions()
                

    def _get_missing_keys(self) -> set:
        return set(self.default_dists.keys()) - set(self.tx_params_dists.keys())
    

    def _validate_distributions(self):
        if self.missing_policy == MissingPolicy.RAISE:
            missing_keys = self._get_missing_keys()
            if missing_keys:
                raise ValueError(
                    f"Missing required parameters: {[key.value for key in missing_keys]}. "
                    f"Current policy is RAISE. Use fill_missing() to change policy."
                )
    
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

        if self.missing_policy == MissingPolicy.CONSTANTS:
            sampled_params.update(self.constant_values)

        elif self.missing_policy == MissingPolicy.IGNORE:
            for key in self._get_missing_keys():
                sampled_params[key] = 0.0

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

    def reset(self) -> None:
        """Reset class paramters to inital values"""
        self.__init__(self._initial_sample_rate, self._initial_tx_params_distributions, self._initial_seed)

    def clone(self, seed: int | None = None):
        return ADSBEncoder(
            sample_rate=self.sample_rate,
            tx_params_distributions=self.tx_params_dists,
            seed=seed if seed is not None else self._seed,
        )