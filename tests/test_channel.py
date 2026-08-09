import pytest

from src.adsb_generator.channel import ADSBChannel, ChannelParams


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


class TestADSBChannelInit:
    def test_default_distributions(self):
        ch = ADSBChannel()
        expected = {
            ChannelParams.SNR_DB: [
                [3.0, 8.0, 0.20],
                [8.0, 15.0, 0.55],
                [15.0, 25.0, 0.25],
            ],
            ChannelParams.NOISE_CORRELATION: [
                [0.0, 0.0, 1.0],
            ],
            ChannelParams.FREQUENCY_OFFSET: [
                [-1000.0, 1000.0, 0.80],
                [-3000.0, 3000.0, 0.20],
            ],
            ChannelParams.PHASE_OFFSET: [
                [-0.10, 0.10, 0.80],
                [-0.30, 0.30, 0.20],
            ],
            ChannelParams.DC_OFFSET_I: [
                [-0.01, 0.01, 0.90],
                [-0.03, 0.03, 0.10],
            ],
            ChannelParams.DC_OFFSET_Q: [
                [-0.01, 0.01, 0.90],
                [-0.03, 0.03, 0.10],
            ],
            ChannelParams.IQ_GAIN_IMBALANCE: [
                [0.00, 0.02, 0.80],
                [0.02, 0.05, 0.20],
            ],
            ChannelParams.IQ_PHASE_IMBALANCE: [
                [-1.0, 1.0, 0.80],
                [-3.0, 3.0, 0.20],
            ],
        }
        assert ch.channel_params_dists == expected

    def test_custom_distributions(self):
        custom = {
            ChannelParams.SNR_DB: [[0.0, 0.5, 0.6], [0.5, 1.0, 0.4]],
        }
        ch = ADSBChannel(channel_params_distributions=custom)
        assert ch.channel_params_dists == custom

    def test_seed_from_parameter(self):
        ch = ADSBChannel(seed=42)
        ch2 = ADSBChannel(seed=42)
        assert ch._seed == 42
        assert ch2._seed == 42

    def test_reproducible_rng_with_same_seed(self):
        ch1 = ADSBChannel(seed=123)
        ch2 = ADSBChannel(seed=123)
        v1 = ch1._rng.random()
        v2 = ch2._rng.random()
        assert v1 == v2

    def test_different_seeds_different_rng(self):
        ch1 = ADSBChannel(seed=1)
        ch2 = ADSBChannel(seed=2)
        v1 = ch1._rng.random()
        v2 = ch2._rng.random()
        assert v1 != v2

    def test_default_seed_is_random(self):
        ch1 = ADSBChannel()
        ch2 = ADSBChannel()
        assert ch1._seed != ch2._seed

    def test_seed_property(self):
        ch = ADSBChannel(seed=99)
        assert ch._seed == 99


class TestValidateDistributions:
    def test_default_distributions_are_valid(self):
        ch = ADSBChannel()
        ch._validate_distributions()

    def test_custom_valid_distributions_are_accepted(self):
        custom = {
            ChannelParams.SNR_DB: [[0.0, 0.5, 0.5], [0.5, 1.0, 0.5]],
        }
        ch = ADSBChannel(channel_params_distributions=custom)
        ch._validate_distributions()

    def test_rejects_invalid_key_by_enum(self):
        custom = {"INVALID": [[0.0, 1.0, 1.0]]}
        ch = ADSBChannel(channel_params_distributions=custom)
        with pytest.raises(ValueError, match="Invalid channel param key"):
            ch._validate_distributions()

    def test_rejects_invalid_key_by_string(self):
        custom = {"bad_key": [[0.0, 1.0, 1.0]]}
        ch = ADSBChannel(channel_params_distributions=custom)
        with pytest.raises(ValueError, match="Invalid channel param key"):
            ch._validate_distributions()

    def test_accepts_string_key_snr_db(self):
        custom = {"snr_db": [[0.0, 0.5, 0.5], [0.5, 1.0, 0.5]]}
        ch = ADSBChannel(channel_params_distributions=custom)
        ch._validate_distributions()

    def test_accepts_string_key_frequency_offset(self):
        custom = {"frequency_offset": [[0.0, 0.5, 0.5], [0.5, 1.0, 0.5]]}
        ch = ADSBChannel(channel_params_distributions=custom)
        ch._validate_distributions()

    def test_rejects_min_greater_than_max(self):
        custom = {
            ChannelParams.SNR_DB: [[0.5, 0.0, 1.0]],
        }
        ch = ADSBChannel(channel_params_distributions=custom)
        with pytest.raises(ValueError, match="Invalid range"):
            ch._validate_distributions()

    def test_rejects_weights_summing_too_low(self):
        custom = {
            ChannelParams.SNR_DB: [[0.0, 1.0, 0.3]],
        }
        ch = ADSBChannel(channel_params_distributions=custom)
        with pytest.raises(ValueError, match="Sum of weights"):
            ch._validate_distributions()

    def test_rejects_weights_summing_too_high(self):
        custom = {
            ChannelParams.SNR_DB: [[0.0, 0.5, 0.6], [0.5, 1.0, 0.6]],
        }
        ch = ADSBChannel(channel_params_distributions=custom)
        with pytest.raises(ValueError, match="Sum of weights"):
            ch._validate_distributions()

    def test_boundary_weight_sum_099_is_accepted(self):
        custom = {
            ChannelParams.SNR_DB: [[0.0, 1.0, 0.99]],
        }
        ch = ADSBChannel(channel_params_distributions=custom)
        ch._validate_distributions()

    def test_boundary_weight_sum_101_is_accepted(self):
        custom = {
            ChannelParams.SNR_DB: [[0.0, 1.0, 1.01]],
        }
        ch = ADSBChannel(channel_params_distributions=custom)
        ch._validate_distributions()


class TestSampleChannelParams:
    def setup_method(self):
        self.ch = ADSBChannel(seed=42)

    def test_returns_dict(self):
        result = self.ch._sample_channel_params()
        assert isinstance(result, dict)

    def test_keys_are_channelparams_enums(self):
        result = self.ch._sample_channel_params()
        for key in result:
            assert isinstance(key, ChannelParams)

    def test_contains_all_expected_params(self):
        result = self.ch._sample_channel_params()
        assert ChannelParams.SNR_DB in result
        assert ChannelParams.NOISE_CORRELATION in result
        assert ChannelParams.FREQUENCY_OFFSET in result
        assert ChannelParams.PHASE_OFFSET in result
        assert ChannelParams.DC_OFFSET_I in result
        assert ChannelParams.DC_OFFSET_Q in result
        assert ChannelParams.IQ_GAIN_IMBALANCE in result
        assert ChannelParams.IQ_PHASE_IMBALANCE in result

    def test_values_are_floats(self):
        result = self.ch._sample_channel_params()
        for val in result.values():
            assert isinstance(val, float)

    def test_snr_db_in_expected_range(self):
        for _ in range(100):
            result = self.ch._sample_channel_params()
            assert 3.0 <= result[ChannelParams.SNR_DB] <= 25.0

    def test_noise_correlation_is_zero(self):
        for _ in range(100):
            result = self.ch._sample_channel_params()
            assert result[ChannelParams.NOISE_CORRELATION] == 0.0

    def test_frequency_offset_in_expected_range(self):
        for _ in range(100):
            result = self.ch._sample_channel_params()
            assert -3000.0 <= result[ChannelParams.FREQUENCY_OFFSET] <= 3000.0

    def test_phase_offset_in_expected_range(self):
        for _ in range(100):
            result = self.ch._sample_channel_params()
            assert -0.30 <= result[ChannelParams.PHASE_OFFSET] <= 0.30

    def test_dc_offset_i_in_expected_range(self):
        for _ in range(100):
            result = self.ch._sample_channel_params()
            assert -0.03 <= result[ChannelParams.DC_OFFSET_I] <= 0.03

    def test_dc_offset_q_in_expected_range(self):
        for _ in range(100):
            result = self.ch._sample_channel_params()
            assert -0.03 <= result[ChannelParams.DC_OFFSET_Q] <= 0.03

    def test_iq_gain_imbalance_in_expected_range(self):
        for _ in range(100):
            result = self.ch._sample_channel_params()
            assert 0.00 <= result[ChannelParams.IQ_GAIN_IMBALANCE] <= 0.05

    def test_iq_phase_imbalance_in_expected_range(self):
        for _ in range(100):
            result = self.ch._sample_channel_params()
            assert -3.0 <= result[ChannelParams.IQ_PHASE_IMBALANCE] <= 3.0

    def test_reproducible_with_same_seed(self):
        ch1 = ADSBChannel(seed=99)
        ch2 = ADSBChannel(seed=99)
        assert ch1._sample_channel_params() == ch2._sample_channel_params()

    def test_different_seeds_different_result(self):
        ch1 = ADSBChannel(seed=1)
        ch2 = ADSBChannel(seed=2)
        assert ch1._sample_channel_params() != ch2._sample_channel_params()

    def test_multiple_calls_produce_varied_snr(self):
        snrs = {self.ch._sample_channel_params()[ChannelParams.SNR_DB] for _ in range(50)}
        assert len(snrs) > 1

    def test_multiple_calls_produce_varied_frequency_offset(self):
        offsets = {
            self.ch._sample_channel_params()[ChannelParams.FREQUENCY_OFFSET] for _ in range(50)
        }
        assert len(offsets) > 1

    def test_string_keys_produce_channelparams_enum_keys(self):
        custom = {
            "snr_db": [[0.1, 0.2, 1.0]],
            "frequency_offset": [[0.1, 0.2, 1.0]],
        }
        ch = ADSBChannel(channel_params_distributions=custom, seed=42)
        result = ch._sample_channel_params()
        for key in result:
            assert isinstance(key, ChannelParams)
        assert ChannelParams.SNR_DB in result
        assert ChannelParams.FREQUENCY_OFFSET in result

    def test_values_respect_narrow_custom_range(self):
        custom = {
            ChannelParams.SNR_DB: [[10.0, 10.1, 1.0]],
            ChannelParams.FREQUENCY_OFFSET: [[500.0, 510.0, 1.0]],
        }
        ch = ADSBChannel(channel_params_distributions=custom, seed=42)
        for _ in range(100):
            result = ch._sample_channel_params()
            assert 10.0 <= result[ChannelParams.SNR_DB] <= 10.1
            assert 500.0 <= result[ChannelParams.FREQUENCY_OFFSET] <= 510.0
