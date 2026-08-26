from enum import Enum

class MessageType(Enum):
    """Supported Automatic Dependent Surveillance-Broadcast (ADS-B) message types."""
    IDENTIFICATION = "identification"
    SURFACE_POSITION = "surface_position"
    AIRBORNE_POSITION = "airborne_position"
    AIRBORNE_VELOCITY = "airborne_velocity"

class TXParams(Enum):
    AMPLITUDE = "amplitude"

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