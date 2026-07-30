import random
import numpy as np
from enum import Enum

class TXParams(Enum):
    AMPLITUDE = "amplitude"

class ADSBEncoder:
    def __init__(self, sample_rate: float = 2e6, tx_params_distributions: dict | None = None, seed: int | None = None):
        self.sample_rate = sample_rate

        default_dists = {
                TXParams.AMPLITUDE: [
                    [0.05, 0.25, 0.5],
                    [0.25, 0.65, 0.3],
                    [0.65, 1.00, 0.2]
                ]}

        self.tx_params_dists = tx_params_distributions or default_dists

        self._validate_distributions

        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._rng = random.Random(self._seed)

    @property
    def seed(self) -> int:
        """Gets the seed value passed at initialization."""
        return self._seed

    def _validate_distributions(self):
        valid_types = set(TXParams)
        valid_values = {item.value for item in TXParams}

        for tx_param, intervals in self.tx_params_dists.items():
            if tx_param not in valid_types and tx_param not in valid_values:
                raise ValueError(
                    f"Invalid tx param key: '{tx_param}'. "
                    f"Valid options are the TXParams enums or: {valid_values}"
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

            enum_key = param_key if isinstance(param_key, TXParams) else TXParams(param_key)
            sampled_params[enum_key] = sampled_val

        return sampled_params

    def encode(self, msg: int) -> tuple[np.ndarray, dict[TXParams, float]]:
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