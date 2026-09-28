"""Runtime configuration tests for ADSBTransmitter, ADSBChannel, and ADSBMessage."""

import pytest
from src.adsb_generator.transmitter import ADSBTransmitter
from src.adsb_generator.channel import ADSBChannel
from src.adsb_generator.message import ADSBMessage
from src.adsb_generator.types import TXParams, ChannelParams, MessageType


class TestConfigureTransmitter:
    def setup_method(self):
        self.enc = ADSBTransmitter(seed=42)

    def test_returns_none(self):
        result = self.enc.configure()
        assert result is None

    def test_updates_sample_rate(self):
        self.enc.configure(sample_rate=4e6)
        assert self.enc.sample_rate == 4e6

    def test_updates_tx_params(self):
        new_params = {
            TXParams.AMPLITUDE: [[0.5, 0.5, 1.0]],
            TXParams.AMPLITUDE_DROOP: [[0.5, 0.5, 1.0]],
            TXParams.PHASE_NOISE_LEVEL: [[0.5, 0.5, 1.0]],
            TXParams.PHASE_NOISE_BANDWIDTH: [[0.5, 0.5, 1.0]],
        }
        self.enc.configure(tx_params_distributions=new_params)
        assert self.enc.tx_params_dists == new_params

    def test_changes_seed(self):
        self.enc.configure(seed=99)
        assert self.enc.seed == 99

    def test_reseeds_rng(self):
        self.enc.configure(seed=99)
        sample_a = self.enc._sample_tx_params()
        self.enc.configure(seed=99)
        sample_b = self.enc._sample_tx_params()
        assert sample_a == sample_b

    def test_no_args_preserves_state(self):
        original_rate = self.enc.sample_rate
        original_params = dict(self.enc.tx_params_dists)
        self.enc.configure()
        assert self.enc.sample_rate == original_rate
        assert self.enc.tx_params_dists == original_params

    def test_seed_only_preserves_others(self):
        original_rate = self.enc.sample_rate
        original_params = dict(self.enc.tx_params_dists)
        self.enc.configure(seed=99)
        assert self.enc.sample_rate == original_rate
        assert self.enc.tx_params_dists == original_params

    def test_sample_rate_only_preserves_seed_and_params(self):
        original_seed = self.enc.seed
        original_params = dict(self.enc.tx_params_dists)
        self.enc.configure(sample_rate=4e6)
        assert self.enc.seed == original_seed
        assert self.enc.tx_params_dists == original_params

    def test_validates_rejects_bad_prob_after_update(self):
        bad = {
            TXParams.AMPLITUDE: [[0.0, 0.5, 0.6], [0.5, 1.0, 0.6]],
        }
        with pytest.raises(ValueError, match="Sum of weights"):
            self.enc.configure(tx_params_distributions=bad)

    def test_validates_rejects_bad_key_after_update(self):
        with pytest.raises(ValueError, match="Invalid tx param key"):
            self.enc.configure(tx_params_distributions={"bad_key": [[0.0, 1.0, 1.0]]})

    def test_sample_rate_affects_transmit(self):
        self.enc.configure(sample_rate=4e6)
        clean, _, _ = self.enc.transmit(0)
        expected = int(round(120.0 * 4e6 / 1e6))
        assert len(clean) == expected

class TestConfigureChannel:
    def setup_method(self):
        self.ch = ADSBChannel(seed=42)

    def test_returns_none(self):
        result = self.ch.configure()
        assert result is None

    def test_updates_sample_rate(self):
        self.ch.configure(sample_rate=4e6)
        assert self.ch.sample_rate == 4e6

    def test_updates_channel_params(self):
        new_params = {
            ChannelParams.SNR_DB: [[10.0, 10.1, 1.0]],
            ChannelParams.FREQUENCY_OFFSET: [[500.0, 510.0, 1.0]],
            ChannelParams.PHASE_OFFSET: [[-0.10, 0.10, 1.0]],
            ChannelParams.NOISE_CORRELATION: [[0.0, 0.0, 1.0]],
            ChannelParams.IQ_GAIN_IMBALANCE: [[0.00, 0.02, 1.0]],
            ChannelParams.IQ_PHASE_IMBALANCE: [[-1.0, 1.0, 1.0]],
            ChannelParams.DC_OFFSET_I: [[-0.01, 0.01, 1.0]],
            ChannelParams.DC_OFFSET_Q: [[-0.01, 0.01, 1.0]],
        }
        self.ch.configure(channel_params_distributions=new_params)
        assert self.ch.channel_params_dists == new_params

    def test_changes_seed(self):
        self.ch.configure(seed=99)
        assert self.ch.seed == 99

    def test_reseeds_rng(self):
        self.ch.configure(seed=99)
        sample_a = self.ch._sample_channel_params()
        self.ch.configure(seed=99)
        sample_b = self.ch._sample_channel_params()
        assert sample_a == sample_b

    def test_no_args_preserves_state(self):
        original_rate = self.ch.sample_rate
        original_params = dict(self.ch.channel_params_dists)
        self.ch.configure()
        assert self.ch.sample_rate == original_rate
        assert self.ch.channel_params_dists == original_params

    def test_seed_only_preserves_others(self):
        original_rate = self.ch.sample_rate
        original_params = dict(self.ch.channel_params_dists)
        self.ch.configure(seed=99)
        assert self.ch.sample_rate == original_rate
        assert self.ch.channel_params_dists == original_params

    def test_sample_rate_only_preserves_seed_and_params(self):
        original_seed = self.ch.seed
        original_params = dict(self.ch.channel_params_dists)
        self.ch.configure(sample_rate=4e6)
        assert self.ch.seed == original_seed
        assert self.ch.channel_params_dists == original_params

    def test_validates_rejects_bad_weights_after_update(self):
        bad = {
            ChannelParams.SNR_DB: [[0.0, 0.5, 0.6], [0.5, 1.0, 0.6]],
        }
        with pytest.raises(ValueError, match="Sum of weights"):
            self.ch.configure(channel_params_distributions=bad)

    def test_validates_rejects_bad_key_after_update(self):
        with pytest.raises(ValueError, match="Invalid channel param key"):
            self.ch.configure(channel_params_distributions={"bad_key": [[0.0, 1.0, 1.0]]})

class TestConfigureMessage:
    def setup_method(self):
        self.msg = ADSBMessage(seed=42)

    def test_returns_none(self):
        result = self.msg.configure()
        assert result is None

    def test_updates_probs(self):
        new_probs = {
            MessageType.IDENTIFICATION: 1.0,
            MessageType.SURFACE_POSITION: 0.0,
            MessageType.AIRBORNE_POSITION: 0.0,
            MessageType.AIRBORNE_VELOCITY: 0.0,
        }
        self.msg.configure(message_type_probs=new_probs)
        assert self.msg.message_type_probs[MessageType.IDENTIFICATION] == 1.0

    def test_replaces_all_probs(self):
        new_probs = {
            MessageType.IDENTIFICATION: 0.5,
            MessageType.SURFACE_POSITION: 0.2,
            MessageType.AIRBORNE_POSITION: 0.2,
            MessageType.AIRBORNE_VELOCITY: 0.1,
        }
        self.msg.configure(message_type_probs=new_probs)
        assert self.msg.message_type_probs == new_probs

    def test_no_args_preserves_state(self):
        original_probs = dict(self.msg.message_type_probs)
        self.msg.configure()
        assert self.msg.message_type_probs == original_probs

    def test_changes_seed(self):
        self.msg.configure(seed=99)
        assert self.msg.seed == 99

    def test_reseeds_rng(self):
        self.msg.configure(seed=99)
        sample_a = self.msg.build()
        self.msg.configure(seed=99)
        sample_b = self.msg.build()
        assert sample_a == sample_b

    def test_seed_only_preserves_probs(self):
        original_probs = dict(self.msg.message_type_probs)
        self.msg.configure(seed=99)
        assert self.msg.message_type_probs == original_probs

    def test_probs_only_preserves_seed(self):
        original_seed = self.msg.seed
        new_probs = {
            MessageType.IDENTIFICATION: 0.5,
            MessageType.SURFACE_POSITION: 0.2,
            MessageType.AIRBORNE_POSITION: 0.2,
            MessageType.AIRBORNE_VELOCITY: 0.1,
        }
        self.msg.configure(message_type_probs=new_probs)
        assert self.msg.seed == original_seed

    def test_validates_rejects_bad_prob_after_update(self):
        bad = {
            MessageType.IDENTIFICATION: 0.5,
            MessageType.SURFACE_POSITION: 0.5,
            MessageType.AIRBORNE_POSITION: 0.5,
            MessageType.AIRBORNE_VELOCITY: 0.5,
        }
        with pytest.raises(ValueError, match="Sum of probabilities"):
            self.msg.configure(message_type_probs=bad)

    def test_validates_rejects_bad_key_after_update(self):
        with pytest.raises(ValueError, match="Invalid message type key"):
            self.msg.configure(message_type_probs={"bad_type": 1.0})

    def test_reconfigure_makes_build_use_new_probs(self):
        ident_only = {
            MessageType.IDENTIFICATION: 1.0,
            MessageType.SURFACE_POSITION: 0.0,
            MessageType.AIRBORNE_POSITION: 0.0,
            MessageType.AIRBORNE_VELOCITY: 0.0,
        }
        self.msg.configure(message_type_probs=ident_only)
        for _ in range(50):
            _, msg_type = self.msg.build()
            assert msg_type == MessageType.IDENTIFICATION
