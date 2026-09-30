import numpy as np
from enum import Enum
from dataclasses import dataclass

class MessageType(Enum):
    """Supported Automatic Dependent Surveillance-Broadcast (ADS-B) message types."""
    IDENTIFICATION = "identification"
    SURFACE_POSITION = "surface_position"
    AIRBORNE_POSITION = "airborne_position"
    AIRBORNE_VELOCITY = "airborne_velocity"

class TXParams(Enum):
    """Supported transmitter paramaters"""
    AMPLITUDE = "amplitude"
    AMPLITUDE_DROOP = "amplitude_droop"
    PHASE_NOISE_LEVEL = "phase_noise_level"
    PHASE_NOISE_BANDWIDTH = "phase_noise_bandwidth"

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

class MissingPolicy(Enum):
    RAISE = "raise"
    IGNORE = "ignore"
    DEFAULTS = "defaults"
    CONSTANTS = "constant"

@dataclass
class ADSBSample:
    """A single ADS-B sample containing the raw message, encoded signal, and impaired signal."""
    message: int
    message_type: MessageType

    clean_signal: np.ndarray
    transmitted_signal: np.ndarray
    tx_params: dict[TXParams, float]

    channel_signal: np.ndarray
    channel_params: dict[ChannelParams, float]