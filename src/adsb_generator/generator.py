import random
import numpy as np
from dataclasses import dataclass
from .message import ADSBMessage, MessageType
from .encoder import ADSBEncoder, TXParams
from .channel import ADSBChannel, ChannelParams

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
    def __init__(self, message_type_probs: dict[MessageType | str, float] | None = None , tx_params_distributions: dict[TXParams | str, list[list[float]]] | None = None,
                 channel_params_distributions: dict[ChannelParams | str, list[list[float]]] | None = None, sample_rate: float = 2e6, seed: int | None = None):
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

        self.builder = ADSBMessage(message_type_probs, self._seed)
        self.encoder = ADSBEncoder(sample_rate, tx_params_distributions, self._seed)
        self.channel = ADSBChannel(sample_rate, channel_params_distributions, self._seed)

    @property
    def seed(self) -> int:
        """Gets the seed value."""
        return self._seed

    def __iter__(self):
        """
        Yields an infinite stream of ADS-B samples.

        Each iteration builds a random ADS-B message, encodes it into a baseband
        I/Q signal, applies channel impairments, and yields an ADSBSample containing
        the raw message, clean signal, and impaired signal along with their parameters.
        """
        while True:
            message, message_type = self.builder.build()

            clean_signal, tx_params = self.encoder.encode(message)

            channel_signal, channel_params = self.channel.apply(clean_signal)

            yield ADSBSample(
                message=message,
                message_type=message_type,
                clean_signal=clean_signal,
                tx_params=tx_params,
                channel_signal=channel_signal,
                channel_params=channel_params,
            )