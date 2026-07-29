import random
from enum import Enum

class TXParams(Enum):
    AMPLITUDE = "amplitude"
    THRESHOLD = "threshold"

class ADSBEncoder:
    def __init__(self, sample_rate: float = 2e6, tx_params_distributions: dict | None = None, seed: int | None = None):
        self.sample_rate = sample_rate

        default_dists = {
                TXParams.AMPLITUDE: [
                    [0.05, 0.25, 0.5],
                    [0.25, 0.65, 0.3],
                    [0.65, 1.00, 0.2]
                ],
                TXParams.THRESHOLD: [
                    [0.02, 0.10, 0.5],
                    [0.10, 0.35, 0.3],
                    [0.35, 0.60, 0.2]
            ]}

        self.tx_params_dists = tx_params_distributions or default_dists

        self._validate_distributions

        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._rng = random.Random(self._seed)

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

        