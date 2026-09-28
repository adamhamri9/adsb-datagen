import numpy as np
import pytest
from src.adsb_generator.transmitter import ADSBTransmitter
from src.adsb_generator.types import TXParams


class TestADSBTransmitterInit:
    def test_default_sample_rate(self):
        enc = ADSBTransmitter()
        assert enc.sample_rate == 2e6

    def test_custom_sample_rate(self):
        enc = ADSBTransmitter(sample_rate=1e6)
        assert enc.sample_rate == 1e6

    def test_default_distributions(self):
        enc = ADSBTransmitter()
        expected = {
            TXParams.AMPLITUDE: [
                [0.05, 0.25, 0.5],
                [0.25, 0.65, 0.3],
                [0.65, 1.00, 0.2]
            ],
            TXParams.AMPLITUDE_DROOP: [
                [0.00, 0.02, 0.5],
                [0.02, 0.05, 0.35],
                [0.05, 0.10, 0.15]
            ],
            TXParams.PHASE_NOISE_LEVEL: [
                [0.001, 0.003, 0.2],
                [0.003, 0.010, 0.6],
                [0.010, 0.020, 0.2]
            ],
            TXParams.PHASE_NOISE_BANDWIDTH: [
                [1000.0, 10000.0, 0.2],
                [10000.0, 100.0e3, 0.6],
                [100.0e3, 500.0e3, 0.2]
            ]}
        
        assert enc.tx_params_dists == expected

    def test_custom_distributions(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 0.5, 0.6], [0.5, 1.0, 0.4]],
        }
        enc = ADSBTransmitter(tx_params_distributions=custom)
        assert enc.tx_params_dists == custom

    def test_seed_from_parameter(self):
        enc = ADSBTransmitter(seed=42)
        enc2 = ADSBTransmitter(seed=42)
        assert enc._seed == 42
        assert enc2._seed == 42

    def test_reproducible_rng_with_same_seed(self):
        enc1 = ADSBTransmitter(seed=123)
        enc2 = ADSBTransmitter(seed=123)
        v1 = enc1._rng.random()
        v2 = enc2._rng.random()
        assert v1 == v2

    def test_different_seeds_different_rng(self):
        enc1 = ADSBTransmitter(seed=1)
        enc2 = ADSBTransmitter(seed=2)
        v1 = enc1._rng.random()
        v2 = enc2._rng.random()
        assert v1 != v2

    def test_default_seed_is_random(self):
        enc1 = ADSBTransmitter()
        enc2 = ADSBTransmitter()
        assert enc1._seed != enc2._seed

    def test_seed_property(self):
        enc = ADSBTransmitter(seed=99)
        assert enc._seed == 99


class TestValidateDistributions:
    def test_default_distributions_are_valid(self):
        enc = ADSBTransmitter()
        enc._validate_distributions()

    def test_custom_valid_distributions_are_accepted(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 0.5, 0.5], [0.5, 1.0, 0.5]],
        }
        enc = ADSBTransmitter(tx_params_distributions=custom)
        enc._validate_distributions()

    def test_rejects_invalid_key_by_enum(self):
        custom = {"INVALID": [[0.0, 1.0, 1.0]]}
        with pytest.raises(ValueError, match="Invalid tx param key"):
            ADSBTransmitter(tx_params_distributions=custom)

    def test_rejects_invalid_key_by_string(self):
        custom = {"bad_key": [[0.0, 1.0, 1.0]]}
        with pytest.raises(ValueError, match="Invalid tx param key"):
            ADSBTransmitter(tx_params_distributions=custom)

    def test_rejects_min_greater_than_max(self):
        custom = {
            TXParams.AMPLITUDE: [[0.5, 0.0, 1.0]],
        }
        with pytest.raises(ValueError, match="Invalid range"):
            ADSBTransmitter(tx_params_distributions=custom)

    def test_rejects_weights_summing_too_low(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 1.0, 0.3]],
        }
        with pytest.raises(ValueError, match="Sum of weights"):
            ADSBTransmitter(tx_params_distributions=custom)

    def test_rejects_weights_summing_too_high(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 0.5, 0.6], [0.5, 1.0, 0.6]],
        }
        with pytest.raises(ValueError, match="Sum of weights"):
            ADSBTransmitter(tx_params_distributions=custom)

    def test_boundary_weight_sum_099_is_accepted(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 1.0, 0.99]],
        }
        enc = ADSBTransmitter(tx_params_distributions=custom)
        enc._validate_distributions()

    def test_boundary_weight_sum_101_is_accepted(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 1.0, 1.01]],
        }
        enc = ADSBTransmitter(tx_params_distributions=custom)
        enc._validate_distributions()


class TestSampleTxParams:
    def setup_method(self):
        self.enc = ADSBTransmitter(seed=42)

    def test_returns_dict(self):
        result = self.enc._sample_tx_params()
        assert isinstance(result, dict)

    def test_keys_are_txparams_enums(self):
        result = self.enc._sample_tx_params()
        for key in result:
            assert isinstance(key, TXParams)

    def test_contains_amplitude(self):
        result = self.enc._sample_tx_params()
        assert TXParams.AMPLITUDE in result

    def test_values_are_floats(self):
        result = self.enc._sample_tx_params()
        for val in result.values():
            assert isinstance(val, float)

    def test_amplitude_in_expected_range(self):
        for _ in range(100):
            result = self.enc._sample_tx_params()
            assert 0.05 <= result[TXParams.AMPLITUDE] <= 1.0

    def test_reproducible_with_same_seed(self):
        enc1 = ADSBTransmitter(seed=99)
        enc2 = ADSBTransmitter(seed=99)
        assert enc1._sample_tx_params() == enc2._sample_tx_params()

    def test_different_seeds_different_result(self):
        enc1 = ADSBTransmitter(seed=1)
        enc2 = ADSBTransmitter(seed=2)
        assert enc1._sample_tx_params() != enc2._sample_tx_params()

    def test_multiple_calls_produce_varied_values(self):
        amplitudes = {self.enc._sample_tx_params()[TXParams.AMPLITUDE] for _ in range(50)}
        assert len(amplitudes) > 1

    def test_values_respect_narrow_custom_range(self):
        custom = {
            TXParams.AMPLITUDE: [[0.75, 0.80, 1.0]],
        }
        enc = ADSBTransmitter(tx_params_distributions=custom, seed=42)
        for _ in range(100):
            result = enc._sample_tx_params()
            assert 0.75 <= result[TXParams.AMPLITUDE] <= 0.80


class TestApplyAmplitudeDroop:
    def setup_method(self):
        self.tx = ADSBTransmitter(seed=42)

    def test_returns_ndarray(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.1)
        assert isinstance(result, np.ndarray)

    def test_returns_complex_dtype(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.1)
        assert np.iscomplexobj(result)

    def test_preserves_shape(self):
        signal = np.ones(64, dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.1)
        assert result.shape == signal.shape

    def test_zero_droop_preserves_shape(self):
        signal = np.ones(64, dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.0)
        assert result.shape == signal.shape

    def test_zero_droop_returns_unchanged_signal(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.0)
        np.testing.assert_array_equal(result, signal)

    def test_negative_droop_returns_unchanged_signal(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, -0.5)
        np.testing.assert_array_equal(result, signal)

    def test_first_sample_is_unchanged(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.5)
        assert result[0] == pytest.approx(signal[0])

    def test_last_sample_scaled_by_one_minus_droop(self):
        signal = np.ones(10, dtype=np.complex128)
        droop = 0.25
        result = self.tx._apply_amplitude_droop(signal, droop)
        assert result[-1] == pytest.approx(signal[-1] * (1.0 - droop))

    def test_matches_linear_ramp(self):
        signal = np.ones(10, dtype=np.complex128)
        droop = 0.5
        result = self.tx._apply_amplitude_droop(signal, droop)
        expected = signal * np.linspace(1.0, 1.0 - droop, 10)
        np.testing.assert_allclose(result, expected)

    def test_known_values(self):
        signal = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.6)
        expected = np.array([1.0, 0.8, 0.6, 0.4], dtype=np.complex128)
        np.testing.assert_allclose(result, expected)

    def test_amplitude_decreases_monotonically(self):
        signal = np.ones(32, dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.1)
        assert np.all(np.diff(result.real) <= 0)

    def test_full_droop_zeroes_last_sample(self):
        signal = np.ones(10, dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 1.0)
        assert result[-1] == pytest.approx(0.0)

    def test_does_not_mutate_input(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex128)
        original = signal.copy()
        self.tx._apply_amplitude_droop(signal, 0.5)
        np.testing.assert_array_equal(signal, original)

    def test_zero_samples_stay_zero(self):
        signal = np.zeros(10, dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.5)
        np.testing.assert_allclose(result, signal)

    def test_scales_magnitude_proportionally(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)
        droop = 0.5
        result = self.tx._apply_amplitude_droop(signal, droop)
        ratio = result / signal
        expected = np.linspace(1.0, 1.0 - droop, 4)
        np.testing.assert_allclose(ratio, expected)

    def test_preserves_imaginary_part_sign(self):
        signal = np.array([1.0 + 2.0j, 1.0 - 2.0j], dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.5)
        assert np.all(np.sign(result.imag) == np.sign(signal.imag))

    def test_larger_droop_gives_more_attenuation(self):
        signal = np.ones(16, dtype=np.complex128)
        small = self.tx._apply_amplitude_droop(signal, 0.1)
        large = self.tx._apply_amplitude_droop(signal, 0.5)
        assert large[-1].real < small[-1].real

    def test_works_with_real_dtype_input(self):
        signal = np.ones(10, dtype=np.float64)
        result = self.tx._apply_amplitude_droop(signal, 0.5)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, np.linspace(1.0, 0.5, 10))

    def test_works_with_empty_signal(self):
        signal = np.array([], dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.5)
        assert result.shape == (0,)

    def test_single_sample_signal(self):
        signal = np.array([2.0], dtype=np.complex128)
        result = self.tx._apply_amplitude_droop(signal, 0.5)
        assert result[0] == pytest.approx(2.0)

    def test_real_transmit_signal_length(self):
        clean, transmitted, params = self.tx.transmit(0)
        assert transmitted.shape == clean.shape
        drooped = self.tx._apply_amplitude_droop(clean, params[TXParams.AMPLITUDE_DROOP])
        assert drooped.shape == clean.shape


class TestApplyPhaseNoise:
    def setup_method(self):
        self.tx = ADSBTransmitter(seed=42)
        self.signal = np.ones(240, dtype=np.complex64)

    def test_returns_ndarray(self):
        result = self.tx._apply_phase_noise(self.signal, 0.05, 1e4)
        assert isinstance(result, np.ndarray)

    def test_returns_complex_dtype(self):
        result = self.tx._apply_phase_noise(self.signal, 0.05, 1e4)
        assert np.iscomplexobj(result)

    def test_preserves_shape(self):
        result = self.tx._apply_phase_noise(self.signal, 0.05, 1e4)
        assert result.shape == self.signal.shape

    def test_preserves_magnitude(self):
        result = self.tx._apply_phase_noise(self.signal, 0.5, 1e4)
        np.testing.assert_allclose(np.abs(result), np.abs(self.signal), rtol=1e-5)

    def test_preserves_magnitude_for_complex_signal(self):
        signal = (np.arange(240) + 1j * np.arange(240)[::-1]).astype(np.complex64)
        result = self.tx._apply_phase_noise(signal, 0.2, 1e4)
        np.testing.assert_allclose(np.abs(result), np.abs(signal), rtol=1e-5)

    def test_changes_phase(self):
        result = self.tx._apply_phase_noise(self.signal, 0.5, 1e4)
        assert not np.allclose(result, self.signal)

    def test_zero_level_returns_unchanged_signal(self):
        result = self.tx._apply_phase_noise(self.signal, 0.0, 1e4)
        np.testing.assert_array_equal(result, self.signal)

    def test_negative_level_returns_unchanged_signal(self):
        result = self.tx._apply_phase_noise(self.signal, -0.1, 1e4)
        np.testing.assert_array_equal(result, self.signal)

    def test_zero_bandwidth_returns_unchanged_signal(self):
        result = self.tx._apply_phase_noise(self.signal, 0.05, 0.0)
        np.testing.assert_array_equal(result, self.signal)

    def test_negative_bandwidth_returns_unchanged_signal(self):
        result = self.tx._apply_phase_noise(self.signal, 0.05, -1e4)
        np.testing.assert_array_equal(result, self.signal)

    def test_works_with_empty_signal(self):
        signal = np.array([], dtype=np.complex64)
        result = self.tx._apply_phase_noise(signal, 0.05, 1e4)
        assert result.shape == (0,)

    def test_phase_std_matches_level(self):
        level = 0.05
        result = self.tx._apply_phase_noise(self.signal, level, 1e4)
        phase = np.angle(result / self.signal)
        assert np.std(phase) == pytest.approx(level, rel=1e-3)

    def test_larger_level_gives_more_phase_deviation(self):
        small = self.tx._apply_phase_noise(self.signal, 0.01, 1e4)
        large = self.tx._apply_phase_noise(self.signal, 0.2, 1e4)
        p_small = np.angle(small / self.signal)
        p_large = np.angle(large / self.signal)
        assert np.std(p_large) > np.std(p_small)

    def test_bandwidth_above_nyquist_is_clamped(self):
        above = ADSBTransmitter(seed=1)._apply_phase_noise(self.signal, 0.05, 1e9)
        clamped = ADSBTransmitter(seed=1)._apply_phase_noise(
            self.signal, 0.05, 0.499 * self.tx.sample_rate
        )
        np.testing.assert_array_equal(above, clamped)

    def test_larger_bandwidth_is_less_smooth(self):
        low = ADSBTransmitter(seed=7)._apply_phase_noise(self.signal, 0.05, 100.0)
        high = ADSBTransmitter(seed=7)._apply_phase_noise(self.signal, 0.05, 1e5)
        assert np.std(np.diff(np.angle(low))) < np.std(np.diff(np.angle(high)))

    def test_does_not_mutate_input(self):
        signal = np.array([1.0 + 2.0j, -3.0 + 4.0j], dtype=np.complex64)
        original = signal.copy()
        self.tx._apply_phase_noise(signal, 0.5, 1e4)
        np.testing.assert_array_equal(signal, original)

    def test_zero_samples_stay_zero(self):
        signal = np.zeros(240, dtype=np.complex64)
        result = self.tx._apply_phase_noise(signal, 0.5, 1e4)
        np.testing.assert_allclose(result, signal)

    def test_reproducible_with_same_seed(self):
        a = ADSBTransmitter(seed=99)._apply_phase_noise(self.signal, 0.05, 1e4)
        b = ADSBTransmitter(seed=99)._apply_phase_noise(self.signal, 0.05, 1e4)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_different_result(self):
        a = ADSBTransmitter(seed=1)._apply_phase_noise(self.signal, 0.05, 1e4)
        b = ADSBTransmitter(seed=2)._apply_phase_noise(self.signal, 0.05, 1e4)
        assert not np.array_equal(a, b)

    def test_repeated_calls_consume_rng(self):
        first = self.tx._apply_phase_noise(self.signal, 0.05, 1e4)
        second = self.tx._apply_phase_noise(self.signal, 0.05, 1e4)
        assert not np.array_equal(first, second)

    def test_works_with_real_dtype_input(self):
        signal = np.ones(240, dtype=np.float64)
        result = self.tx._apply_phase_noise(signal, 0.05, 1e4)
        assert np.iscomplexobj(result)
        np.testing.assert_allclose(np.abs(result), np.ones(240), rtol=1e-5)

    def test_output_dtype_follows_input_dtype(self):
        result64 = self.tx._apply_phase_noise(self.signal.astype(np.complex64), 0.05, 1e4)
        result128 = self.tx._apply_phase_noise(self.signal.astype(np.complex128), 0.05, 1e4)
        assert result64.dtype == np.complex64
        assert result128.dtype == np.complex128

    def test_works_with_single_sample(self):
        signal = np.array([1.0 + 0.0j], dtype=np.complex64)
        result = self.tx._apply_phase_noise(signal, 0.05, 1e4)
        assert result.shape == (1,)
        np.testing.assert_allclose(np.abs(result), np.ones(1), rtol=1e-5)

    def test_works_with_real_transmit_output(self):
        clean, _, params = self.tx.transmit(0)
        result = self.tx._apply_phase_noise(clean, params[TXParams.PHASE_NOISE_LEVEL], 1e4)
        assert result.shape == clean.shape
        np.testing.assert_allclose(np.abs(result), np.abs(clean), rtol=1e-5)

    def test_high_level_still_preserves_magnitude(self):
        result = self.tx._apply_phase_noise(self.signal, 2.0, 1e4)
        np.testing.assert_allclose(np.abs(result), np.abs(self.signal), rtol=1e-5)


class TestTransmit:
    def setup_method(self):
        self.enc = ADSBTransmitter(seed=42)

    def test_returns_tuple(self):
        result = self.enc.transmit(0)
        assert isinstance(result, tuple)

    def test_tuple_has_three_elements(self):
        result = self.enc.transmit(0)
        assert len(result) == 3

    def test_first_element_is_ndarray(self):
        clean, _, _ = self.enc.transmit(0)
        assert isinstance(clean, np.ndarray)

    def test_second_element_is_ndarray(self):
        _, transmitted, _ = self.enc.transmit(0)
        assert isinstance(transmitted, np.ndarray)

    def test_third_element_is_dict(self):
        _, _, params = self.enc.transmit(0)
        assert isinstance(params, dict)

    def test_iq_dtype_is_complex64(self):
        clean, _, _ = self.enc.transmit(0)
        assert clean.dtype == np.complex64

    def test_default_sample_rate_length(self):
        clean, _, _ = self.enc.transmit(0)
        assert len(clean) == 240

    def test_custom_sample_rate_length(self):
        enc = ADSBTransmitter(sample_rate=1e6, seed=42)
        clean, _, _ = enc.transmit(0)
        assert len(clean) == 120

    def test_imaginary_part_is_zero(self):
        clean, _, _ = self.enc.transmit(0)
        assert np.all(clean.imag == 0.0)

    def test_signal_values_are_zero_or_amplitude(self):
        clean, _, params = self.enc.transmit(0)
        real = clean.real
        amps = np.unique(real)
        for a in amps:
            assert a == 0.0 or a == pytest.approx(params[TXParams.AMPLITUDE])

    def test_returned_params_contains_amplitude(self):
        _, _, params = self.enc.transmit(0)
        assert TXParams.AMPLITUDE in params

    def test_returned_amplitude_is_float(self):
        _, _, params = self.enc.transmit(0)
        assert isinstance(params[TXParams.AMPLITUDE], float)

    def test_preamble_pulses_present_at_expected_positions(self):
        clean, _, params = self.enc.transmit(0)
        amps = params[TXParams.AMPLITUDE]
        real = clean.real

        spus = self.enc.sample_rate / 1e6
        expected_starts = [0.0, 1.0, 3.5, 4.5]
        for start_us in expected_starts:
            idx = int(round(start_us * spus))
            assert real[idx] == pytest.approx(amps), f"preamble pulse missing at {start_us}us (idx {idx})"

    def test_preamble_gaps_are_zero(self):
        clean, _, _ = self.enc.transmit(0)
        real = clean.real

        spus = self.enc.sample_rate / 1e6
        gap_regions = [(0.5, 1.0), (1.5, 3.5), (4.0, 4.5)]
        for start_us, end_us in gap_regions:
            s = int(round(start_us * spus))
            e = int(round(end_us * spus))
            assert np.all(real[s:e] == 0.0), f"gap [{start_us},{end_us})us is not all zeros"

    def test_all_zeros_message_no_first_half_pulses(self):
        for enc_seed in [42, 7, 99]:
            enc = ADSBTransmitter(seed=enc_seed)
            clean, _, params = enc.transmit(0)
            real = clean.real
            spus = enc.sample_rate / 1e6
            for bit_idx in range(112):
                bit_start = 8.0 + bit_idx
                first_half_start = int(round(bit_start * spus))
                first_half_end = int(round((bit_start + 0.5) * spus))
                assert np.all(real[first_half_start:first_half_end] == 0.0), \
                    f"bit {bit_idx}: first half should be zero for msg=0"

    def test_all_ones_message_pulses_in_first_half(self):
        msg = (1 << 112) - 1
        for enc_seed in [42, 7, 99]:
            enc = ADSBTransmitter(seed=enc_seed)
            clean, _, params = enc.transmit(msg)
            real = clean.real
            spus = enc.sample_rate / 1e6
            amp = params[TXParams.AMPLITUDE]
            for bit_idx in range(112):
                bit_start = 8.0 + bit_idx
                first_half_start = int(round(bit_start * spus))
                first_half_end = int(round((bit_start + 0.5) * spus))
                expected = amp
                assert real[first_half_start] == pytest.approx(expected), \
                    f"bit {bit_idx}: first half should be amplitude for msg=all-ones"

    def test_second_half_zeros_when_bit_is_one(self):
        msg = (1 << 112) - 1
        enc = ADSBTransmitter(seed=42)
        clean, _, _ = enc.transmit(msg)
        real = clean.real
        spus = enc.sample_rate / 1e6
        for bit_idx in range(112):
            bit_start = 8.0 + bit_idx
            second_half_start = int(round((bit_start + 0.5) * spus))
            second_half_end = int(round((bit_start + 1.0) * spus))
            assert np.all(real[second_half_start:second_half_end] == 0.0), \
                f"bit {bit_idx}: second half should be zero when bit=1"

    def test_single_bit_set_at_lsb(self):
        msg = 1
        enc = ADSBTransmitter(seed=42)
        clean, _, params = enc.transmit(msg)
        real = clean.real
        spus = enc.sample_rate / 1e6
        amp = params[TXParams.AMPLITUDE]

        last_bit_start = 8.0 + 111.0
        lsb_first_half_s = int(round(last_bit_start * spus))
        lsb_first_half_e = int(round((last_bit_start + 0.5) * spus))
        assert real[lsb_first_half_s] == pytest.approx(amp), \
            "LSB bit (msg=1) should have pulse in first half"

    def test_single_bit_set_at_msb(self):
        msg = 1 << 111
        enc = ADSBTransmitter(seed=42)
        clean, _, params = enc.transmit(msg)
        real = clean.real
        spus = enc.sample_rate / 1e6
        amp = params[TXParams.AMPLITUDE]

        first_bit_start = 8.0
        msb_first_half_s = int(round(first_bit_start * spus))
        assert real[msb_first_half_s] == pytest.approx(amp), \
            "MSB bit should have pulse in first half"

    def test_reproducible_with_same_seed_and_msg(self):
        enc1 = ADSBTransmitter(seed=123)
        enc2 = ADSBTransmitter(seed=123)
        clean1, _, params1 = enc1.transmit(0xDEADBEEF)
        clean2, _, params2 = enc2.transmit(0xDEADBEEF)
        assert np.array_equal(clean1, clean2)
        assert params1 == params2

    def test_different_seeds_different_amplitude(self):
        enc1 = ADSBTransmitter(seed=1)
        enc2 = ADSBTransmitter(seed=2)
        _, _, p1 = enc1.transmit(0)
        _, _, p2 = enc2.transmit(0)
        assert p1[TXParams.AMPLITUDE] != p2[TXParams.AMPLITUDE]

    def test_different_messages_produce_different_signals(self):
        clean1, _, _ = self.enc.transmit(0xAAAAAAAAAAAAAAAAAAAAAAAAAAAA)
        clean2, _, _ = self.enc.transmit(0x5555555555555555555555555555)
        assert not np.array_equal(clean1, clean2)

    def test_signal_length_scales_with_sample_rate(self):
        for rate in [1e6, 2e6, 4e6]:
            enc = ADSBTransmitter(sample_rate=rate, seed=42)
            clean, _, _ = enc.transmit(0)
            expected = int(round(120.0 * rate / 1e6))
            assert len(clean) == expected

    def test_signal_non_negative_real_part(self):
        clean, _, _ = self.enc.transmit(0)
        assert np.all(clean.real >= 0.0)

    def test_known_signal_sum(self):
        custom = {TXParams.AMPLITUDE: [[1.0, 1.0, 1.0]]}
        enc = ADSBTransmitter(tx_params_distributions=custom, seed=0)
        clean, _, params = enc.transmit(0)
        assert params[TXParams.AMPLITUDE] == 1.0

        nonzero = np.count_nonzero(clean.real)
        preamble_pulses = 4
        data_pulses = 112
        assert nonzero == preamble_pulses + data_pulses
