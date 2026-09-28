"""Missing-parameter policy tests for ADSBTransmitter and ADSBChannel."""

import pytest
from src.adsb_generator.transmitter import ADSBTransmitter
from src.adsb_generator.channel import ADSBChannel
from src.adsb_generator.types import MissingPolicy, TXParams, ChannelParams


class TestGetMissingKeysTransmitter:
    def test_all_keys_present(self):
        enc = ADSBTransmitter(seed=42)
        assert enc._get_missing_keys() == set()

    def test_partial_dict(self):
        partial = {TXParams.AMPLITUDE: [[0.5, 1.0, 1.0]]}
        enc = ADSBTransmitter(tx_params_distributions=partial, seed=42)
        missing = enc._get_missing_keys()
        assert TXParams.AMPLITUDE not in missing
        assert len(missing) == 3

    def test_empty_dict_falls_back_to_defaults(self):
        enc = ADSBTransmitter(tx_params_distributions={}, seed=42)
        assert enc._get_missing_keys() == set()

class TestFillMissingTransmitter:
    def setup_method(self):
        self.partial = {TXParams.AMPLITUDE: [[0.5, 1.0, 1.0]]}
        self.enc = ADSBTransmitter(tx_params_distributions=self.partial, seed=42)

    def test_raise_sets_policy(self):
        with pytest.raises(ValueError, match="Missing required parameters"):
            self.enc.fill_missing(MissingPolicy.RAISE)
        assert self.enc.missing_policy == MissingPolicy.RAISE

    def test_ignore_sets_policy(self):
        self.enc.fill_missing(MissingPolicy.IGNORE)
        assert self.enc.missing_policy == MissingPolicy.IGNORE

    def test_defaults_merges_default_dists(self):
        self.enc.fill_missing(MissingPolicy.DEFAULTS)
        assert TXParams.AMPLITUDE in self.enc.tx_params_dists

    def test_defaults_overwrites_existing(self):
        self.enc.fill_missing(MissingPolicy.DEFAULTS)
        assert self.enc.tx_params_dists[TXParams.AMPLITUDE] != [[0.5, 1.0, 1.0]]

    def test_constants_stores_values(self):
        missing = self.enc._get_missing_keys()
        values = {k: 42.0 for k in missing}
        self.enc.fill_missing(MissingPolicy.CONSTANTS, values=values)
        assert self.enc.missing_policy == MissingPolicy.CONSTANTS
        for k in missing:
            assert self.enc.constant_values[k] == 42.0

    def test_sample_constants_returns_constant_values(self):
        missing = self.enc._get_missing_keys()
        values = {k: 99.0 for k in missing}
        self.enc.fill_missing(MissingPolicy.CONSTANTS, values=values)
        params = self.enc._sample_tx_params()
        for k in missing:
            assert params[k] == 99.0

    def test_sample_ignore_returns_zero_for_missing(self):
        self.enc.fill_missing(MissingPolicy.IGNORE)
        params = self.enc._sample_tx_params()
        for k in self.enc._get_missing_keys():
            assert params[k] == 0.0

    def test_sample_defaults_samples_from_defaults(self):
        self.enc.fill_missing(MissingPolicy.DEFAULTS)
        params = self.enc._sample_tx_params()
        assert TXParams.AMPLITUDE in params
        assert isinstance(params[TXParams.AMPLITUDE], float)

class TestGetMissingKeysChannel:
    def test_all_keys_present(self):
        ch = ADSBChannel(seed=42)
        assert ch._get_missing_keys() == set()

    def test_partial_dict(self):
        partial = {ChannelParams.SNR_DB: [[3.0, 8.0, 1.0]]}
        ch = ADSBChannel(channel_params_distributions=partial, seed=42)
        missing = ch._get_missing_keys()
        assert ChannelParams.SNR_DB not in missing
        assert ChannelParams.FREQUENCY_OFFSET in missing
        assert len(missing) == 7

    def test_empty_dict_falls_back_to_defaults(self):
        ch = ADSBChannel(channel_params_distributions={}, seed=42)
        assert ch._get_missing_keys() == set()

class TestFillMissingChannel:
    def setup_method(self):
        self.partial = {ChannelParams.SNR_DB: [[10.0, 10.1, 1.0]]}
        self.ch = ADSBChannel(channel_params_distributions=self.partial, seed=42)

    def test_raise_sets_policy(self):
        with pytest.raises(ValueError, match="Missing required parameters"):
            self.ch.fill_missing(MissingPolicy.RAISE)

        assert self.ch.missing_policy == MissingPolicy.RAISE

    def test_ignore_sets_policy(self):
        self.ch.fill_missing(MissingPolicy.IGNORE)
        assert self.ch.missing_policy == MissingPolicy.IGNORE

    def test_defaults_merges_default_dists(self):
        self.ch.fill_missing(MissingPolicy.DEFAULTS)
        assert ChannelParams.FREQUENCY_OFFSET in self.ch.channel_params_dists
        assert ChannelParams.IQ_GAIN_IMBALANCE in self.ch.channel_params_dists

    def test_defaults_overwrites_existing(self):
        self.ch.fill_missing(MissingPolicy.DEFAULTS)
        assert self.ch.channel_params_dists[ChannelParams.SNR_DB] != [[10.0, 10.1, 1.0]]

    def test_constants_stores_values(self):
        missing = self.ch._get_missing_keys()
        values = {k: 42.0 for k in missing}
        self.ch.fill_missing(MissingPolicy.CONSTANTS, values=values)
        assert self.ch.missing_policy == MissingPolicy.CONSTANTS
        for k in missing:
            assert self.ch.constant_values[k] == 42.0

    def test_constants_missing_value_raises(self):
        missing = self.ch._get_missing_keys()
        values = {k: 1.0 for k in list(missing)[:-1]}
        with pytest.raises(ValueError, match="Missing constant value"):
            self.ch.fill_missing(MissingPolicy.CONSTANTS, values=values)

    def test_sample_constants_returns_constant_values(self):
        missing = self.ch._get_missing_keys()
        values = {k: 99.0 for k in missing}
        self.ch.fill_missing(MissingPolicy.CONSTANTS, values=values)
        params = self.ch._sample_channel_params()
        for k in missing:
            assert params[k] == 99.0

    def test_sample_ignore_returns_zero_for_missing(self):
        self.ch.fill_missing(MissingPolicy.IGNORE)
        params = self.ch._sample_channel_params()
        for k in self.ch._get_missing_keys():
            assert params[k] == 0.0

    def test_sample_defaults_samples_from_defaults(self):
        self.ch.fill_missing(MissingPolicy.DEFAULTS)
        params = self.ch._sample_channel_params()
        assert ChannelParams.FREQUENCY_OFFSET in params
        assert isinstance(params[ChannelParams.FREQUENCY_OFFSET], float)

    def test_raise_with_missing_keys_validates(self):
        with pytest.raises(ValueError, match="Missing required parameters"):
            self.ch.fill_missing(MissingPolicy.RAISE)
