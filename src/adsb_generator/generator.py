import random
import numpy as np
from dataclasses import dataclass
from .message import ADSBMessage
from .encoder import ADSBEncoder
from .channel import ADSBChannel
from .types import MissingPolicy, MessageType, TXParams, ChannelParams

@dataclass
class ADSBSample:
    """A single ADS-B sample containing the raw message, encoded signal, and impaired signal."""
    message: int
    message_type: MessageType

    clean_signal: np.ndarray
    tx_params: dict[TXParams, float]

    channel_signal: np.ndarray
    channel_params: dict[ChannelParams, float]

class ADSBGenerator():
    """
    End-to-end ADS-B signal generator that produces synthetic I/Q samples with channel impairments.

    This class orchestrates the full ADS-B signal generation pipeline: building random
    112-bit ADS-B messages, encoding them into baseband I/Q signals using PPM, and
    applying realistic RF channel impairments. It is implemented as an infinite iterator
    that yields ADSBSample instances on each iteration.

    Attributes:
        seed (int): Seed used for the internal random number generators.
    """
    def __init__(self, message_type_probs: dict[MessageType, float] | None = None , tx_params_distributions: dict[TXParams, list[list[float]]] | None = None,
                 channel_params_distributions: dict[ChannelParams, list[list[float]]] | None = None, sample_rate: float = 2e6, seed: int | None = None):
        """
        Initializes the generator with message, transmission, and channel distributions.

        Args:
            message_type_probs: Mapping of MessageType to their emission probabilities. 
                If None, defaults to an equal 25% distribution across all message types.
            tx_params_distributions: Mapping of TXParams to probability distributions 
                defined as lists of [min_val, max_val, weight] intervals. If None, 
                defaults to a single amplitude distribution.
            channel_params_distributions: Mapping of ChannelParams to probability 
                distributions defined as lists of [min_val, max_val, weight] intervals. 
                If None, defaults to distributions covering typical ADS-B reception 
                conditions.
            sample_rate: Sampling rate in samples per second. Defaults to 2 MHz.
            seed: Seed for the internal random number generators to ensure reproducible 
                output. If None, the generator is seeded from system randomness.
        """

        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.sample_rate = sample_rate
        self._initial_sample_rate = sample_rate

        self.builder = ADSBMessage(message_type_probs, self._seed)
        self.encoder = ADSBEncoder(sample_rate, tx_params_distributions, self._seed)
        self.channel = ADSBChannel(sample_rate, channel_params_distributions, self._seed)

        self._buffer: list[ADSBSample] = []
        self._buffering = False

    @property
    def seed(self) -> int:
        """Gets the seed value."""
        return self._seed

    def configure(self, message_type_probs: dict[MessageType, float] | None = None , tx_params_distributions: dict[TXParams, list[list[float]]] | None = None,
                 channel_params_distributions: dict[ChannelParams, list[list[float]]] | None = None, sample_rate: float  | None = None, seed: int | None = None, update_initial: bool = False) -> None:
        """Update ADSBMessage, ADSBEncoder, and ADSBChannel configurations."""
        self.builder.configure(message_type_probs, seed, update_initial)
        self.encoder.configure(sample_rate, tx_params_distributions, seed, update_initial)
        self.channel.configure(sample_rate, channel_params_distributions, seed, update_initial)
        self.sample_rate = sample_rate
        self._initial_sample_rate = sample_rate if update_initial else self._initial_sample_rate

    def fill_missing(self, policy: MissingPolicy, tx_values: dict[TXParams, float] | None = None, channel_values: dict[ChannelParams, float] | None = None):
        """
        Configure the missing tx parameters & channel parameters handling policy.

        Args:
            policy: The missing policy to apply (RAISE, IGNORE, DEFAULTS, or CONSTANTS).
            tx_values: Required when policy is CONSTANTS. Maps each TXParams key to its constant value.
            channel_values: Required when policy is CONSTANTS. Maps each ChannelParams key to its constant value.
        """
        self.encoder.fill_missing(policy, tx_values)
        self.channel.fill_missing(policy, channel_values)

    def generate(self, n: int = 1) -> list[ADSBSample] | None:
        samples = [] if not self._buffering else None

        for _ in range(n):
            message, message_type = self.builder.build()

            clean_signal, tx_params = self.encoder.encode(message)

            channel_signal, channel_params = self.channel.apply(clean_signal)

            sample = ADSBSample(
                message=message,
                message_type=message_type,
                clean_signal=clean_signal,
                tx_params=tx_params,
                channel_signal=channel_signal,
                channel_params=channel_params,
            )

            if self._buffering:
                self._buffer.append(sample)
            else:
                samples.append(sample)

        return samples

    def __iter__(self):
        """
        Yields an infinite stream of ADS-B samples.

        Each iteration builds a random ADS-B message, encodes it into a baseband
        I/Q signal, applies channel impairments, and yields an ADSBSample containing
        the raw message, clean signal, and impaired signal along with their parameters.
        """
        return self

    def __next__(self):
        return self.generate()[0]

    def export(self, path: str, samples: list[ADSBSample] | None = None) -> None:
        from pathlib import Path

        path = Path(path)
        data = samples if samples is not None else self._buffer

        if path.suffix == ".npz":
            data_dict = {}
            for i, sample in enumerate(data):
                data_dict[f"sample_{i}"] = self._sample_to_dict(sample)
            np.savez(path, **data_dict)

        elif path.suffix == ".npy":
            if len(data) == 1:
                np.save(path, data[0].clean_signal)
            else:
                raise ValueError(".npy supports only one sample. Use .npz for multiple samples.")

        else:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                "Supported formats: .npz, .npy"
            )


    def _sample_to_dict(self, sample: ADSBSample) -> dict:
        return {
            "message": sample.message,
            "message_type": sample.message_type.value,
            "clean_signal": sample.clean_signal,   
            "channel_signal": sample.channel_signal, 
            "tx_params": sample.tx_params,
            "channel_params": sample.channel_params,
        }
     
    def start_buffering(self) -> None:
        """Enable buffering of generated samples."""
        self._buffering = True

    def stop_buffering(self, clear: bool = True) -> None:
        """
        Disable buffering and optionally clear the buffer.

        Args:
            clear: If True, clear the buffer. Defaults to True.
        """
        self._buffering = False
        if clear:
            self._buffer.clear()

    def reset(self) -> None:
        self.sample_rate = self._initial_sample_rate
        self.builder.reset()
        self.encoder.reset()
        self.channel.reset()

    def clone(self, seed: int | None = None):
        return ADSBGenerator(
            message_type_probs=self.builder.message_type_probs,
            tx_params_distributions=self.encoder.tx_params_dists,
            channel_params_distributions=self.channel.channel_params_dists,
            sample_rate=self.sample_rate,
            seed=seed if seed is not None else self._seed,
            )