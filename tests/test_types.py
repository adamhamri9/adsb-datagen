"""Enum definition tests for TXParams, ChannelParams, and MessageType."""

import numpy as np
from src.adsb_generator.types import TXParams, ChannelParams, MessageType, ADSBSample


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

    
class TestADSBSample:
    def test_has_message_field(self):
        sample = ADSBSample(
            message=123,
            message_type=MessageType.AIRBORNE_POSITION,
            clean_signal=np.array([], dtype=np.complex64),
            transmitted_signal=np.array([], dtype=np.complex64),
            tx_params={TXParams.AMPLITUDE: 1.0},
            channel_signal=np.array([], dtype=np.complex64),
            channel_params={ChannelParams.SNR_DB: 15.0},
        )
        assert sample.message == 123

    def test_has_message_type_field(self):
        sample = ADSBSample(
            message=0,
            message_type=MessageType.AIRBORNE_VELOCITY,
            clean_signal=np.array([], dtype=np.complex64),
            transmitted_signal=np.array([], dtype=np.complex64),
            tx_params={},
            channel_signal=np.array([], dtype=np.complex64),
            channel_params={},
        )
        assert sample.message_type == MessageType.AIRBORNE_VELOCITY

    def test_has_clean_signal(self):
        sig = np.ones(10, dtype=np.complex64)
        sample = ADSBSample(
            message=0,
            message_type=MessageType.IDENTIFICATION,
            clean_signal=sig,
            transmitted_signal=np.array([], dtype=np.complex64),
            tx_params={},
            channel_signal=np.array([], dtype=np.complex64),
            channel_params={},
        )
        np.testing.assert_array_equal(sample.clean_signal, sig)

    def test_has_channel_signal(self):
        sig = np.ones(10, dtype=np.complex64)
        sample = ADSBSample(
            message=0,
            message_type=MessageType.IDENTIFICATION,
            clean_signal=np.array([], dtype=np.complex64),
            transmitted_signal=np.array([], dtype=np.complex64),
            tx_params={},
            channel_signal=sig,
            channel_params={},
        )
        np.testing.assert_array_equal(sample.channel_signal, sig)

    def test_has_tx_params(self):
        tx = {TXParams.AMPLITUDE: 0.5}
        sample = ADSBSample(
            message=0,
            message_type=MessageType.IDENTIFICATION,
            clean_signal=np.array([], dtype=np.complex64),
            transmitted_signal=np.array([], dtype=np.complex64),
            tx_params=tx,
            channel_signal=np.array([], dtype=np.complex64),
            channel_params={},
        )
        assert sample.tx_params == tx

    def test_has_channel_params(self):
        ch = {ChannelParams.SNR_DB: 20.0}
        sample = ADSBSample(
            message=0,
            message_type=MessageType.IDENTIFICATION,
            clean_signal=np.array([], dtype=np.complex64),
            transmitted_signal=np.array([], dtype=np.complex64),
            tx_params={},
            channel_signal=np.array([], dtype=np.complex64),
            channel_params=ch,
        )
        assert sample.channel_params == ch

    def test_has_transmitted_signal(self):
        sig = np.ones(10, dtype=np.complex64)
        sample = ADSBSample(
            message=0,
            message_type=MessageType.IDENTIFICATION,
            clean_signal=np.array([], dtype=np.complex64),
            transmitted_signal=sig,
            tx_params={},
            channel_signal=np.array([], dtype=np.complex64),
            channel_params={},
        )
        np.testing.assert_array_equal(sample.transmitted_signal, sig)