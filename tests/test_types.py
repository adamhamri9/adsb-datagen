"""Enum definition tests for TXParams, ChannelParams, and MessageType."""

from src.adsb_generator.types import TXParams, ChannelParams, MessageType


class TestTXParams:
    def test_has_amplitude(self):
        assert TXParams.AMPLITUDE.value == "amplitude"

    def test_has_amplitude_droop(self):
        assert TXParams.AMPLITUDE_DROOP.value == "amplitude_droop"

    def test_has_phase_noise_level(self):
        assert TXParams.PHASE_NOISE_LEVEL.value == "phase_noise_level"

    def test_has_phase_noise_bandwidth(self):
        assert TXParams.PHASE_NOISE_BANDWIDTH.value == "phase_noise_bandwidth"

    def test_all_params_are_unique(self):
        values = [t.value for t in TXParams]
        assert len(values) == len(set(values))

class TestChannelParams:
    def test_has_snr_db(self):
        assert ChannelParams.SNR_DB.value == "snr_db"

    def test_has_noise_correlation(self):
        assert ChannelParams.NOISE_CORRELATION.value == "noise_correlation"

    def test_has_frequency_offset(self):
        assert ChannelParams.FREQUENCY_OFFSET.value == "frequency_offset"

    def test_has_phase_offset(self):
        assert ChannelParams.PHASE_OFFSET.value == "phase_offset"

    def test_has_dc_offset_i(self):
        assert ChannelParams.DC_OFFSET_I.value == "dc_offset_i"

    def test_has_dc_offset_q(self):
        assert ChannelParams.DC_OFFSET_Q.value == "dc_offset_q"

    def test_has_iq_gain_imbalance(self):
        assert ChannelParams.IQ_GAIN_IMBALANCE.value == "iq_gain_imbalance"

    def test_has_iq_phase_imbalance(self):
        assert ChannelParams.IQ_PHASE_IMBALANCE.value == "iq_phase_imbalance"

    def test_all_params_are_unique(self):
        values = [t.value for t in ChannelParams]
        assert len(values) == len(set(values))

class TestMessageType:
    def test_has_identification(self):
        assert MessageType.IDENTIFICATION.value == "identification"

    def test_has_surface_position(self):
        assert MessageType.SURFACE_POSITION.value == "surface_position"

    def test_has_airborne_position(self):
        assert MessageType.AIRBORNE_POSITION.value == "airborne_position"

    def test_has_airborne_velocity(self):
        assert MessageType.AIRBORNE_VELOCITY.value == "airborne_velocity"

    def test_all_types_are_unique(self):
        values = [t.value for t in MessageType]
        assert len(values) == len(set(values))
