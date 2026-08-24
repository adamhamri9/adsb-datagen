import numpy as np
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
        with pytest.raises(ValueError, match="Invalid channel param key"):
            ADSBChannel(channel_params_distributions=custom)

    def test_rejects_invalid_key_by_string(self):
        custom = {"bad_key": [[0.0, 1.0, 1.0]]}
        with pytest.raises(ValueError, match="Invalid channel param key"):
            ADSBChannel(channel_params_distributions=custom)

    def test_rejects_min_greater_than_max(self):
        custom = {
            ChannelParams.SNR_DB: [[0.5, 0.0, 1.0]],
        }
        with pytest.raises(ValueError, match="Invalid range"):
            ADSBChannel(channel_params_distributions=custom)

    def test_rejects_weights_summing_too_low(self):
        custom = {
            ChannelParams.SNR_DB: [[0.0, 1.0, 0.3]],
        }
        with pytest.raises(ValueError, match="Sum of weights"):
            ADSBChannel(channel_params_distributions=custom)

    def test_rejects_weights_summing_too_high(self):
        custom = {
            ChannelParams.SNR_DB: [[0.0, 0.5, 0.6], [0.5, 1.0, 0.6]],
        }
        with pytest.raises(ValueError, match="Sum of weights"):
            ADSBChannel(channel_params_distributions=custom)

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


    def test_omitted_params_receive_defaults(self):
        custom = {
            ChannelParams.SNR_DB: [[10.0, 10.1, 1.0]],
        }
        ch = ADSBChannel(channel_params_distributions=custom, seed=42)
        result = ch._sample_channel_params()
        assert len(result) == len(ChannelParams)
        assert result[ChannelParams.NOISE_CORRELATION] == 0.0
        assert result[ChannelParams.FREQUENCY_OFFSET] == 0.0
        assert result[ChannelParams.PHASE_OFFSET] == 0.0
        assert result[ChannelParams.DC_OFFSET_I] == 0.0
        assert result[ChannelParams.DC_OFFSET_Q] == 0.0
        assert result[ChannelParams.IQ_GAIN_IMBALANCE] == 0.0
        assert result[ChannelParams.IQ_PHASE_IMBALANCE] == 0.0

    def test_default_snr_db_when_omitted(self):
        custom = {
            ChannelParams.FREQUENCY_OFFSET: [[0.0, 1.0, 1.0]],
        }
        ch = ADSBChannel(channel_params_distributions=custom, seed=42)
        result = ch._sample_channel_params()
        assert result[ChannelParams.SNR_DB] == 15.0

    def test_omitted_param_defaults_are_not_sampled(self):
        custom = {
            ChannelParams.SNR_DB: [[5.0, 5.1, 1.0]],
        }
        ch = ADSBChannel(channel_params_distributions=custom, seed=42)
        defaults_seen = set()
        for _ in range(50):
            result = ch._sample_channel_params()
            defaults_seen.add(result[ChannelParams.DC_OFFSET_I])
        assert len(defaults_seen) == 1
        assert defaults_seen.pop() == 0.0
class TestApplyDCOffset:
    def setup_method(self):
        self.ch = ADSBChannel(seed=42)

    def test_returns_ndarray(self):
        signal = np.zeros(10, dtype=np.complex128)
        result = self.ch._apply_dc_offset(signal, 0.01, -0.02)
        assert isinstance(result, np.ndarray)

    def test_returns_complex_dtype(self):
        signal = np.zeros(10, dtype=np.complex128)
        result = self.ch._apply_dc_offset(signal, 0.01, -0.02)
        assert np.iscomplexobj(result)

    def test_preserves_shape(self):
        signal = np.zeros((4, 32), dtype=np.complex128)
        result = self.ch._apply_dc_offset(signal, 0.01, -0.02)
        assert result.shape == signal.shape

    def test_zero_offset_returns_unchanged_signal(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_dc_offset(signal, 0.0, 0.0)
        np.testing.assert_array_equal(result, signal)

    def test_adds_dc_offset_i_to_real_part(self):
        signal = np.zeros(10, dtype=np.complex128)
        result = self.ch._apply_dc_offset(signal, 0.01, 0.0)
        np.testing.assert_allclose(result.real, np.full(10, 0.01))
        np.testing.assert_allclose(result.imag, np.zeros(10))

    def test_adds_dc_offset_q_to_imaginary_part(self):
        signal = np.zeros(10, dtype=np.complex128)
        result = self.ch._apply_dc_offset(signal, 0.0, -0.02)
        np.testing.assert_allclose(result.real, np.zeros(10))
        np.testing.assert_allclose(result.imag, np.full(10, -0.02))

    def test_adds_both_offsets(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        result = self.ch._apply_dc_offset(signal, 0.5, -0.25)
        expected = np.array([1.5 + 1.75j, -2.5 + 3.75j], dtype=np.complex128)
        np.testing.assert_array_equal(result, expected)

    def test_preserves_signal_values(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 6.0j], dtype=np.complex128)
        result = self.ch._apply_dc_offset(signal, 0.01, 0.02)
        expected = signal + (0.01 + 0.02j)
        np.testing.assert_allclose(result, expected)

    def test_does_not_mutate_input(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        original = signal.copy()
        self.ch._apply_dc_offset(signal, 0.5, -0.25)
        np.testing.assert_array_equal(signal, original)

    def test_works_with_real_dtype_input(self):
        signal = np.zeros(10, dtype=np.float64)
        result = self.ch._apply_dc_offset(signal, 0.01, 0.02)
        assert np.iscomplexobj(result)
        np.testing.assert_allclose(result.real, np.full(10, 0.01))
        np.testing.assert_allclose(result.imag, np.full(10, 0.02))

    def test_works_with_empty_signal(self):
        signal = np.array([], dtype=np.complex128)
        result = self.ch._apply_dc_offset(signal, 0.01, 0.02)
        assert result.shape == (0,)
        assert result.dtype == np.complex128


class TestApplyFreqOffset:
    def setup_method(self):
        self.ch = ADSBChannel(seed=42)

    def test_returns_ndarray(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        assert isinstance(result, np.ndarray)

    def test_returns_complex_dtype(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        assert np.iscomplexobj(result)

    def test_preserves_shape(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        assert result.shape == signal.shape

    def test_zero_offset_returns_unchanged_signal(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        result = self.ch._apply_freq_offset(signal, 0.0, 1000.0)
        np.testing.assert_array_equal(result, signal)

    def test_first_sample_is_unchanged(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        assert result[0] == signal[0]

    def test_preserves_magnitude(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        np.testing.assert_allclose(np.abs(result), np.ones(10))

    def test_rotates_by_expected_phase(self):
        signal = np.ones(5, dtype=np.complex128)
        result = self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        expected_phase = 2 * np.pi * 100.0 * np.arange(5) / 1000.0
        np.testing.assert_allclose(np.angle(result), expected_phase)

    def test_rotates_by_expected_phase_90_degrees(self):
        signal = np.ones(3, dtype=np.complex128)
        result = self.ch._apply_freq_offset(signal, 250.0, 1000.0)
        expected_phase = 2 * np.pi * 250.0 * np.arange(3) / 1000.0
        np.testing.assert_allclose(np.angle(result), expected_phase)

    def test_negative_offset_rotates_opposite_direction(self):
        signal = np.ones(10, dtype=np.complex128)
        positive = self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        negative = self.ch._apply_freq_offset(signal, -100.0, 1000.0)
        np.testing.assert_allclose(negative, np.conj(positive))

    def test_scale_with_sample_rate(self):
        signal = np.ones(5, dtype=np.complex128)
        low_rate = self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        high_rate = self.ch._apply_freq_offset(signal, 100.0, 2000.0)
        np.testing.assert_allclose(
            np.angle(high_rate), 0.5 * np.angle(low_rate)
        )

    def test_applies_to_signal_values(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        result = self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        expected = signal * np.exp(1j * 2 * np.pi * 100.0 * np.arange(2) / 1000.0)
        np.testing.assert_allclose(result, expected)

    def test_does_not_mutate_input(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        original = signal.copy()
        self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        np.testing.assert_array_equal(signal, original)

    def test_works_with_real_dtype_input(self):
        signal = np.ones(5, dtype=np.float64)
        result = self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        assert np.iscomplexobj(result)
        expected_phase = 2 * np.pi * 100.0 * np.arange(5) / 1000.0
        np.testing.assert_allclose(np.angle(result), expected_phase)

    def test_works_with_empty_signal(self):
        signal = np.array([], dtype=np.complex128)
        result = self.ch._apply_freq_offset(signal, 100.0, 1000.0)
        assert result.shape == (0,)
        assert result.dtype == np.complex128


class TestApplyPhaseOffset:
    def setup_method(self):
        self.ch = ADSBChannel(seed=42)

    def test_returns_ndarray(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_phase_offset(signal, 0.5)
        assert isinstance(result, np.ndarray)

    def test_returns_complex_dtype(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_phase_offset(signal, 0.5)
        assert np.iscomplexobj(result)

    def test_preserves_shape(self):
        signal = np.ones((4, 32), dtype=np.complex128)
        result = self.ch._apply_phase_offset(signal, 0.5)
        assert result.shape == signal.shape

    def test_zero_offset_returns_unchanged_signal(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        result = self.ch._apply_phase_offset(signal, 0.0)
        np.testing.assert_array_equal(result, signal)

    def test_preserves_magnitude(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        result = self.ch._apply_phase_offset(signal, 0.5)
        np.testing.assert_allclose(np.abs(result), np.abs(signal))

    def test_rotates_by_expected_phase(self):
        signal = np.ones(5, dtype=np.complex128)
        result = self.ch._apply_phase_offset(signal, 0.5)
        np.testing.assert_allclose(np.angle(result), np.full(5, 0.5))

    def test_rotates_by_pi_over_two(self):
        signal = np.array([1.0 + 0.0j, 0.0 + 1.0j], dtype=np.complex128)
        result = self.ch._apply_phase_offset(signal, np.pi / 2)
        expected = np.array([0.0 + 1.0j, -1.0 + 0.0j], dtype=np.complex128)
        np.testing.assert_allclose(result, expected)

    def test_negative_offset_rotates_opposite_direction(self):
        signal = np.ones(10, dtype=np.complex128)
        positive = self.ch._apply_phase_offset(signal, 0.5)
        negative = self.ch._apply_phase_offset(signal, -0.5)
        np.testing.assert_allclose(negative, np.conj(positive))

    def test_applies_to_signal_values(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        result = self.ch._apply_phase_offset(signal, 0.5)
        expected = signal * np.exp(1j * 0.5)
        np.testing.assert_allclose(result, expected)

    def test_does_not_mutate_input(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        original = signal.copy()
        self.ch._apply_phase_offset(signal, 0.5)
        np.testing.assert_array_equal(signal, original)

    def test_works_with_real_dtype_input(self):
        signal = np.ones(5, dtype=np.float64)
        result = self.ch._apply_phase_offset(signal, 0.5)
        assert np.iscomplexobj(result)
        np.testing.assert_allclose(np.angle(result), np.full(5, 0.5))

    def test_works_with_empty_signal(self):
        signal = np.array([], dtype=np.complex128)
        result = self.ch._apply_phase_offset(signal, 0.5)
        assert result.shape == (0,)
        assert result.dtype == np.complex128


class TestApplyIQImbalance:
    def setup_method(self):
        self.ch = ADSBChannel(seed=42)

    def test_returns_ndarray(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_iq_imbalance(signal, 0.0, 0.0)
        assert isinstance(result, np.ndarray)

    def test_returns_complex_dtype(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_iq_imbalance(signal, 0.0, 0.0)
        assert np.iscomplexobj(result)

    def test_preserves_shape(self):
        signal = np.ones((4, 32), dtype=np.complex128)
        result = self.ch._apply_iq_imbalance(signal, 0.0, 0.0)
        assert result.shape == signal.shape

    def test_zero_imbalance_returns_unchanged_signal(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        result = self.ch._apply_iq_imbalance(signal, 0.0, 0.0)
        np.testing.assert_array_equal(result, signal)

    def test_gain_only_increases_real_decreases_imag(self):
        signal = np.array([1.0 + 1.0j], dtype=np.complex128)
        result = self.ch._apply_iq_imbalance(signal, 0.2, 0.0)
        expected = np.array([1.1 + 0.9j], dtype=np.complex128)
        np.testing.assert_allclose(result, expected)

    def test_gain_only_symmetry(self):
        signal = np.array([1.0 + 1.0j, -1.0 + 1.0j], dtype=np.complex128)
        result = self.ch._apply_iq_imbalance(signal, 0.4, 0.0)
        expected = np.array([1.2 + 0.8j, -1.2 + 0.8j], dtype=np.complex128)
        np.testing.assert_allclose(result, expected)

    def test_phase_only_cross_coupling(self):
        signal = np.array([1.0 + 0.0j], dtype=np.complex128)
        result = self.ch._apply_iq_imbalance(signal, 0.0, 90.0)
        expected = np.array([0.0 + 1.0j], dtype=np.complex128)
        np.testing.assert_allclose(result, expected)

    def test_phase_only_imag_to_real(self):
        signal = np.array([0.0 + 1.0j], dtype=np.complex128)
        result = self.ch._apply_iq_imbalance(signal, 0.0, 90.0)
        expected = np.array([1.0 + 0.0j], dtype=np.complex128)
        np.testing.assert_allclose(result, expected)

    def test_both_imbalances_applied(self):
        signal = np.array([1.0 + 1.0j], dtype=np.complex128)
        result = self.ch._apply_iq_imbalance(signal, 0.2, 90.0)
        expected = np.array([0.9 + 1.1j], dtype=np.complex128)
        np.testing.assert_allclose(result, expected)

    def test_applies_to_signal_values(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        result = self.ch._apply_iq_imbalance(signal, 0.2, 90.0)
        expected = np.array([1.8 + 1.1j, 3.6 - 3.3j], dtype=np.complex128)
        np.testing.assert_allclose(result, expected)

    def test_does_not_mutate_input(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        original = signal.copy()
        self.ch._apply_iq_imbalance(signal, 0.2, 90.0)
        np.testing.assert_array_equal(signal, original)

    def test_works_with_real_dtype_input(self):
        signal = np.ones(5, dtype=np.float64)
        result = self.ch._apply_iq_imbalance(signal, 0.2, 0.0)
        assert np.iscomplexobj(result)
        np.testing.assert_allclose(result.real, np.full(5, 1.1))

    def test_works_with_empty_signal(self):
        signal = np.array([], dtype=np.complex128)
        result = self.ch._apply_iq_imbalance(signal, 0.2, 90.0)
        assert result.shape == (0,)
        assert result.dtype == np.complex128


class TestApplyGaussianNoise:
    def setup_method(self):
        self.ch = ADSBChannel(seed=42)

    def test_returns_ndarray(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_gaussian_noise(signal, 20.0)
        assert isinstance(result, np.ndarray)

    def test_returns_complex_dtype(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_gaussian_noise(signal, 20.0)
        assert np.iscomplexobj(result)

    def test_preserves_shape(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch._apply_gaussian_noise(signal, 20.0)
        assert result.shape == signal.shape

    def test_zero_signal_returns_zeros(self):
        signal = np.zeros(10, dtype=np.complex128)
        result = self.ch._apply_gaussian_noise(signal, 20.0)
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_high_snr_close_to_original(self):
        signal = np.ones(100, dtype=np.complex128)
        result = self.ch._apply_gaussian_noise(signal, 60.0)
        np.testing.assert_allclose(result, signal, atol=0.1)

    def test_low_snr_deviates_from_original(self):
        signal = np.ones(100, dtype=np.complex128)
        result = self.ch._apply_gaussian_noise(signal, 0.0)
        assert not np.allclose(result, signal, atol=0.5)

    def test_noise_power_matches_expected_snr(self):
        signal = np.ones(10000, dtype=np.complex128)
        snr_db = 20.0
        snr_linear = 10 ** (snr_db / 10)
        result = self.ch._apply_gaussian_noise(signal, snr_db)
        noise = result - signal
        measured_noise_power = np.mean(np.abs(noise) ** 2)
        expected_noise_power = 1.0 / snr_linear
        np.testing.assert_allclose(measured_noise_power, expected_noise_power, rtol=0.15)

    def test_additive_noise(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        result = self.ch._apply_gaussian_noise(signal, 20.0)
        noise = result - signal
        assert noise.shape == signal.shape
        assert np.iscomplexobj(noise)

    def test_does_not_mutate_input(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        original = signal.copy()
        self.ch._apply_gaussian_noise(signal, 20.0)
        np.testing.assert_array_equal(signal, original)

    def test_reproducible_with_same_seed(self):
        ch1 = ADSBChannel(seed=99)
        ch2 = ADSBChannel(seed=99)
        signal = np.ones(50, dtype=np.complex128)
        r1 = ch1._apply_gaussian_noise(signal, 10.0)
        r2 = ch2._apply_gaussian_noise(signal, 10.0)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_different_noise(self):
        ch1 = ADSBChannel(seed=1)
        ch2 = ADSBChannel(seed=2)
        signal = np.ones(50, dtype=np.complex128)
        r1 = ch1._apply_gaussian_noise(signal, 10.0)
        r2 = ch2._apply_gaussian_noise(signal, 10.0)
        assert not np.array_equal(r1, r2)

    def test_works_with_real_dtype_input(self):
        signal = np.ones(100, dtype=np.float64)
        result = self.ch._apply_gaussian_noise(signal, 20.0)
        assert np.iscomplexobj(result)
        assert result.shape == signal.shape

    def test_works_with_empty_signal(self):
        signal = np.array([], dtype=np.complex128)
        result = self.ch._apply_gaussian_noise(signal, 20.0)
        assert result.shape == (0,)
        assert result.dtype == np.complex128

    def test_noise_iq_correlated_with_correlation(self):
        ch = ADSBChannel(seed=42)
        signal = np.ones(10000, dtype=np.complex128)
        result = ch._apply_gaussian_noise(signal, 10.0, noise_correlation=0.8)
        noise = result - signal
        corr = np.corrcoef(noise.real, noise.imag)[0, 1]
        assert corr > 0.5

    def test_noise_iq_uncorrelated_without_correlation(self):
        ch = ADSBChannel(seed=42)
        signal = np.ones(10000, dtype=np.complex128)
        result = ch._apply_gaussian_noise(signal, 10.0, noise_correlation=0)
        noise = result - signal
        corr = np.corrcoef(noise.real, noise.imag)[0, 1]
        assert abs(corr) < 0.1


class TestApply:
    def setup_method(self):
        self.ch = ADSBChannel(seed=42)

    def test_returns_tuple(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.ch.apply(signal)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_ndarray(self):
        signal = np.ones(10, dtype=np.complex128)
        out_signal, _ = self.ch.apply(signal)
        assert isinstance(out_signal, np.ndarray)

    def test_returns_channelparams_dict(self):
        signal = np.ones(10, dtype=np.complex128)
        _, params = self.ch.apply(signal)
        assert isinstance(params, dict)
        for key in params:
            assert isinstance(key, ChannelParams)

    def test_returns_all_params(self):
        signal = np.ones(10, dtype=np.complex128)
        _, params = self.ch.apply(signal)
        assert len(params) == len(ChannelParams)

    def test_preserves_shape(self):
        signal = np.ones(10, dtype=np.complex128)
        out_signal, _ = self.ch.apply(signal)
        assert out_signal.shape == signal.shape

    def test_returns_complex_dtype(self):
        signal = np.ones(10, dtype=np.complex128)
        out_signal, _ = self.ch.apply(signal)
        assert np.iscomplexobj(out_signal)

    def test_reproducible_with_same_seed(self):
        ch1 = ADSBChannel(seed=99)
        ch2 = ADSBChannel(seed=99)
        signal = np.ones(50, dtype=np.complex128)
        out1, params1 = ch1.apply(signal)
        out2, params2 = ch2.apply(signal)
        np.testing.assert_array_equal(out1, out2)
        assert params1 == params2

    def test_different_seeds_different_output(self):
        ch1 = ADSBChannel(seed=1)
        ch2 = ADSBChannel(seed=2)
        signal = np.ones(50, dtype=np.complex128)
        out1, _ = ch1.apply(signal)
        out2, _ = ch2.apply(signal)
        assert not np.array_equal(out1, out2)

    def test_does_not_mutate_input(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        original = signal.copy()
        self.ch.apply(signal)
        np.testing.assert_array_equal(signal, original)

    def test_zero_impairments_close_to_original(self):
        custom = {
            ChannelParams.SNR_DB: [[60.0, 60.0, 1.0]],
            ChannelParams.NOISE_CORRELATION: [[0.0, 0.0, 1.0]],
            ChannelParams.FREQUENCY_OFFSET: [[0.0, 0.0, 1.0]],
            ChannelParams.PHASE_OFFSET: [[0.0, 0.0, 1.0]],
            ChannelParams.DC_OFFSET_I: [[0.0, 0.0, 1.0]],
            ChannelParams.DC_OFFSET_Q: [[0.0, 0.0, 1.0]],
            ChannelParams.IQ_GAIN_IMBALANCE: [[0.0, 0.0, 1.0]],
            ChannelParams.IQ_PHASE_IMBALANCE: [[0.0, 0.0, 1.0]],
        }
        ch = ADSBChannel(channel_params_distributions=custom, seed=42)
        signal = np.ones(100, dtype=np.complex128)
        out_signal, _ = ch.apply(signal)
        np.testing.assert_allclose(out_signal, signal, atol=0.1)

    def test_default_impairments_change_signal(self):
        signal = np.ones(100, dtype=np.complex128)
        out_signal, _ = self.ch.apply(signal)
        assert not np.allclose(out_signal, signal)

    def test_sample_rate_is_used(self):
        ch_slow = ADSBChannel(sample_rate=1000.0, seed=42)
        ch_fast = ADSBChannel(sample_rate=1e6, seed=42)
        custom = {
            ChannelParams.SNR_DB: [[60.0, 60.0, 1.0]],
            ChannelParams.FREQUENCY_OFFSET: [[100.0, 100.0, 1.0]],
        }
        ch_slow.channel_params_distributions = custom
        ch_fast.channel_params_distributions = custom
        signal = np.ones(100, dtype=np.complex128)
        out_slow, _ = ch_slow.apply(signal)
        out_fast, _ = ch_fast.apply(signal)
        assert not np.array_equal(out_slow, out_fast)

    def test_empty_signal(self):
        signal = np.array([], dtype=np.complex128)
        out_signal, params = self.ch.apply(signal)
        assert out_signal.shape == (0,)
        assert len(params) == len(ChannelParams)

    def test_works_with_real_dtype_input(self):
        signal = np.ones(100, dtype=np.float64)
        out_signal, params = self.ch.apply(signal)
        assert np.iscomplexobj(out_signal)
        assert out_signal.shape == signal.shape
        assert len(params) == len(ChannelParams)

    def test_params_match_sampled_values(self):
        custom = {
            ChannelParams.SNR_DB: [[20.0, 20.0, 1.0]],
            ChannelParams.FREQUENCY_OFFSET: [[50.0, 50.0, 1.0]],
            ChannelParams.PHASE_OFFSET: [[0.1, 0.1, 1.0]],
            ChannelParams.DC_OFFSET_I: [[0.01, 0.01, 1.0]],
            ChannelParams.DC_OFFSET_Q: [[0.02, 0.02, 1.0]],
            ChannelParams.IQ_GAIN_IMBALANCE: [[0.03, 0.03, 1.0]],
            ChannelParams.IQ_PHASE_IMBALANCE: [[0.5, 0.5, 1.0]],
        }
        ch = ADSBChannel(channel_params_distributions=custom, seed=42)
        signal = np.ones(10, dtype=np.complex128)
        _, params = ch.apply(signal)
        assert params[ChannelParams.SNR_DB] == 20.0
        assert params[ChannelParams.FREQUENCY_OFFSET] == 50.0
        assert params[ChannelParams.PHASE_OFFSET] == 0.1
        assert params[ChannelParams.DC_OFFSET_I] == 0.01
        assert params[ChannelParams.DC_OFFSET_Q] == 0.02
        assert params[ChannelParams.IQ_GAIN_IMBALANCE] == 0.03
        assert params[ChannelParams.IQ_PHASE_IMBALANCE] == 0.5

    def test_distortion_order_matters(self):
        custom = {
            ChannelParams.SNR_DB: [[60.0, 60.0, 1.0]],
            ChannelParams.FREQUENCY_OFFSET: [[0.0, 0.0, 1.0]],
            ChannelParams.PHASE_OFFSET: [[0.0, 0.0, 1.0]],
            ChannelParams.NOISE_CORRELATION: [[0.0, 0.0, 1.0]],
            ChannelParams.IQ_GAIN_IMBALANCE: [[0.5, 0.5, 1.0]],
            ChannelParams.IQ_PHASE_IMBALANCE: [[0.0, 0.0, 1.0]],
            ChannelParams.DC_OFFSET_I: [[1.0, 1.0, 1.0]],
            ChannelParams.DC_OFFSET_Q: [[0.0, 0.0, 1.0]],
        }
        ch = ADSBChannel(channel_params_distributions=custom, seed=42)
        signal = np.zeros(10, dtype=np.complex128)
        out_signal, _ = ch.apply(signal)
        np.testing.assert_allclose(out_signal, 1.0 + 0.0j, atol=0.1)


class TestConfigure:
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
