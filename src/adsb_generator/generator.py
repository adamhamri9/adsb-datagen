import random
import numpy as np
from dataclasses import dataclass
from .message import ADSBMessage, MessageType
from .encoder import ADSBEncoder, TXParams
from .channel import ADSBChannel, ChannelParams

@dataclass
class ADSBSample:
    message: int
    message_type: MessageType

    clean_signal: np.ndarray
    tx_params: dict[TXParams, float]

    channel_signal: np.ndarray
    channel_params: dict[ChannelParams, float]

class ADSBGenerator():
    def __init__(self, message_type_probs: dict[MessageType | str, float] | None = None , tx_params_distributions: dict[TXParams | str, list[list[float]]] | None = None,
                 channel_params_distributions: dict[ChannelParams | str, list[list[float]]] | None = None, sample_rate: float = 2e6, seed: int | None = None):

        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        self.builder = ADSBMessage(message_type_probs, self._seed)
        self.encoder = ADSBEncoder(sample_rate, tx_params_distributions, self._seed)
        self.channel = ADSBChannel(sample_rate, channel_params_distributions, self._seed)

    @property
    def seed(self) -> int:
        """Gets the seed value."""
        return self._seed

    def __iter__(self):
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