import numpy as np
import pytest

from src.adsb_generator.encoder import ADSBEncoder, TXParams


class TestTXParams:
    def test_has_amplitude(self):
        assert TXParams.AMPLITUDE.value == "amplitude"

    def test_all_params_are_unique(self):
        values = [t.value for t in TXParams]
        assert len(values) == len(set(values))


class TestADSBEncoderInit:
    def test_default_sample_rate(self):
        enc = ADSBEncoder()
        assert enc.sample_rate == 2e6

    def test_custom_sample_rate(self):
        enc = ADSBEncoder(sample_rate=1e6)
        assert enc.sample_rate == 1e6

    def test_default_distributions(self):
        enc = ADSBEncoder()
        expected = {
            TXParams.AMPLITUDE: [
                [0.05, 0.25, 0.5],
                [0.25, 0.65, 0.3],
                [0.65, 1.00, 0.2],
            ],
        }
        assert enc.tx_params_dists == expected

    def test_custom_distributions(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 0.5, 0.6], [0.5, 1.0, 0.4]],
        }
        enc = ADSBEncoder(tx_params_distributions=custom)
        assert enc.tx_params_dists == custom

    def test_seed_from_parameter(self):
        enc = ADSBEncoder(seed=42)
        enc2 = ADSBEncoder(seed=42)
        assert enc._seed == 42
        assert enc2._seed == 42

    def test_reproducible_rng_with_same_seed(self):
        enc1 = ADSBEncoder(seed=123)
        enc2 = ADSBEncoder(seed=123)
        v1 = enc1._rng.random()
        v2 = enc2._rng.random()
        assert v1 == v2

    def test_different_seeds_different_rng(self):
        enc1 = ADSBEncoder(seed=1)
        enc2 = ADSBEncoder(seed=2)
        v1 = enc1._rng.random()
        v2 = enc2._rng.random()
        assert v1 != v2

    def test_default_seed_is_random(self):
        enc1 = ADSBEncoder()
        enc2 = ADSBEncoder()
        assert enc1._seed != enc2._seed

    def test_seed_property(self):
        enc = ADSBEncoder(seed=99)
        assert enc._seed == 99


class TestValidateDistributions:
    def test_default_distributions_are_valid(self):
        enc = ADSBEncoder()
        enc._validate_distributions()

    def test_custom_valid_distributions_are_accepted(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 0.5, 0.5], [0.5, 1.0, 0.5]],
        }
        enc = ADSBEncoder(tx_params_distributions=custom)
        enc._validate_distributions()

    def test_rejects_invalid_key_by_enum(self):
        custom = {"INVALID": [[0.0, 1.0, 1.0]]}
        with pytest.raises(ValueError, match="Invalid tx param key"):
            ADSBEncoder(tx_params_distributions=custom)

    def test_rejects_invalid_key_by_string(self):
        custom = {"bad_key": [[0.0, 1.0, 1.0]]}
        with pytest.raises(ValueError, match="Invalid tx param key"):
            ADSBEncoder(tx_params_distributions=custom)

    def test_rejects_min_greater_than_max(self):
        custom = {
            TXParams.AMPLITUDE: [[0.5, 0.0, 1.0]],
        }
        with pytest.raises(ValueError, match="Invalid range"):
            ADSBEncoder(tx_params_distributions=custom)

    def test_rejects_weights_summing_too_low(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 1.0, 0.3]],
        }
        with pytest.raises(ValueError, match="Sum of weights"):
            ADSBEncoder(tx_params_distributions=custom)

    def test_rejects_weights_summing_too_high(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 0.5, 0.6], [0.5, 1.0, 0.6]],
        }
        with pytest.raises(ValueError, match="Sum of weights"):
            ADSBEncoder(tx_params_distributions=custom)

    def test_boundary_weight_sum_099_is_accepted(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 1.0, 0.99]],
        }
        enc = ADSBEncoder(tx_params_distributions=custom)
        enc._validate_distributions()

    def test_boundary_weight_sum_101_is_accepted(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 1.0, 1.01]],
        }
        enc = ADSBEncoder(tx_params_distributions=custom)
        enc._validate_distributions()


class TestSampleTxParams:
    def setup_method(self):
        self.enc = ADSBEncoder(seed=42)

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
        enc1 = ADSBEncoder(seed=99)
        enc2 = ADSBEncoder(seed=99)
        assert enc1._sample_tx_params() == enc2._sample_tx_params()

    def test_different_seeds_different_result(self):
        enc1 = ADSBEncoder(seed=1)
        enc2 = ADSBEncoder(seed=2)
        assert enc1._sample_tx_params() != enc2._sample_tx_params()

    def test_multiple_calls_produce_varied_values(self):
        amplitudes = {self.enc._sample_tx_params()[TXParams.AMPLITUDE] for _ in range(50)}
        assert len(amplitudes) > 1

    def test_values_respect_narrow_custom_range(self):
        custom = {
            TXParams.AMPLITUDE: [[0.75, 0.80, 1.0]],
        }
        enc = ADSBEncoder(tx_params_distributions=custom, seed=42)
        for _ in range(100):
            result = enc._sample_tx_params()
            assert 0.75 <= result[TXParams.AMPLITUDE] <= 0.80


class TestEncode:
    def setup_method(self):
        self.enc = ADSBEncoder(seed=42)

    def test_returns_tuple(self):
        result = self.enc.encode(0)
        assert isinstance(result, tuple)

    def test_tuple_has_two_elements(self):
        result = self.enc.encode(0)
        assert len(result) == 2

    def test_first_element_is_ndarray(self):
        iq, _ = self.enc.encode(0)
        assert isinstance(iq, np.ndarray)

    def test_second_element_is_dict(self):
        _, params = self.enc.encode(0)
        assert isinstance(params, dict)

    def test_iq_dtype_is_complex64(self):
        iq, _ = self.enc.encode(0)
        assert iq.dtype == np.complex64

    def test_default_sample_rate_length(self):
        iq, _ = self.enc.encode(0)
        assert len(iq) == 240

    def test_custom_sample_rate_length(self):
        enc = ADSBEncoder(sample_rate=1e6, seed=42)
        iq, _ = enc.encode(0)
        assert len(iq) == 120

    def test_imaginary_part_is_zero(self):
        iq, _ = self.enc.encode(0)
        assert np.all(iq.imag == 0.0)

    def test_signal_values_are_zero_or_amplitude(self):
        iq, params = self.enc.encode(0)
        real = iq.real
        amps = np.unique(real)
        for a in amps:
            assert a == 0.0 or a == pytest.approx(params[TXParams.AMPLITUDE])

    def test_returned_params_contains_amplitude(self):
        _, params = self.enc.encode(0)
        assert TXParams.AMPLITUDE in params

    def test_returned_amplitude_is_float(self):
        _, params = self.enc.encode(0)
        assert isinstance(params[TXParams.AMPLITUDE], float)

    def test_preamble_pulses_present_at_expected_positions(self):
        iq, params = self.enc.encode(0)
        amps = params[TXParams.AMPLITUDE]
        real = iq.real

        spus = self.enc.sample_rate / 1e6
        expected_starts = [0.0, 1.0, 3.5, 4.5]
        for start_us in expected_starts:
            idx = int(round(start_us * spus))
            assert real[idx] == pytest.approx(amps), f"preamble pulse missing at {start_us}us (idx {idx})"

    def test_preamble_gaps_are_zero(self):
        iq, _ = self.enc.encode(0)
        real = iq.real

        spus = self.enc.sample_rate / 1e6
        gap_regions = [(0.5, 1.0), (1.5, 3.5), (4.0, 4.5)]
        for start_us, end_us in gap_regions:
            s = int(round(start_us * spus))
            e = int(round(end_us * spus))
            assert np.all(real[s:e] == 0.0), f"gap [{start_us},{end_us})us is not all zeros"

    def test_all_zeros_message_no_first_half_pulses(self):
        for enc_seed in [42, 7, 99]:
            enc = ADSBEncoder(seed=enc_seed)
            iq, params = enc.encode(0)
            real = iq.real
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
            enc = ADSBEncoder(seed=enc_seed)
            iq, params = enc.encode(msg)
            real = iq.real
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
        enc = ADSBEncoder(seed=42)
        iq, _ = enc.encode(msg)
        real = iq.real
        spus = enc.sample_rate / 1e6
        for bit_idx in range(112):
            bit_start = 8.0 + bit_idx
            second_half_start = int(round((bit_start + 0.5) * spus))
            second_half_end = int(round((bit_start + 1.0) * spus))
            assert np.all(real[second_half_start:second_half_end] == 0.0), \
                f"bit {bit_idx}: second half should be zero when bit=1"

    def test_single_bit_set_at_lsb(self):
        msg = 1
        enc = ADSBEncoder(seed=42)
        iq, params = enc.encode(msg)
        real = iq.real
        spus = enc.sample_rate / 1e6
        amp = params[TXParams.AMPLITUDE]

        last_bit_start = 8.0 + 111.0
        lsb_first_half_s = int(round(last_bit_start * spus))
        lsb_first_half_e = int(round((last_bit_start + 0.5) * spus))
        assert real[lsb_first_half_s] == pytest.approx(amp), \
            "LSB bit (msg=1) should have pulse in first half"

    def test_single_bit_set_at_msb(self):
        msg = 1 << 111
        enc = ADSBEncoder(seed=42)
        iq, params = enc.encode(msg)
        real = iq.real
        spus = enc.sample_rate / 1e6
        amp = params[TXParams.AMPLITUDE]

        first_bit_start = 8.0
        msb_first_half_s = int(round(first_bit_start * spus))
        assert real[msb_first_half_s] == pytest.approx(amp), \
            "MSB bit should have pulse in first half"

    def test_reproducible_with_same_seed_and_msg(self):
        enc1 = ADSBEncoder(seed=123)
        enc2 = ADSBEncoder(seed=123)
        iq1, params1 = enc1.encode(0xDEADBEEF)
        iq2, params2 = enc2.encode(0xDEADBEEF)
        assert np.array_equal(iq1, iq2)
        assert params1 == params2

    def test_different_seeds_different_amplitude(self):
        enc1 = ADSBEncoder(seed=1)
        enc2 = ADSBEncoder(seed=2)
        _, p1 = enc1.encode(0)
        _, p2 = enc2.encode(0)
        assert p1[TXParams.AMPLITUDE] != p2[TXParams.AMPLITUDE]

    def test_different_messages_produce_different_signals(self):
        iq1, _ = self.enc.encode(0xAAAAAAAAAAAAAAAAAAAAAAAAAAAA)
        iq2, _ = self.enc.encode(0x5555555555555555555555555555)
        assert not np.array_equal(iq1, iq2)

    def test_signal_length_scales_with_sample_rate(self):
        for rate in [1e6, 2e6, 4e6]:
            enc = ADSBEncoder(sample_rate=rate, seed=42)
            iq, _ = enc.encode(0)
            expected = int(round(120.0 * rate / 1e6))
            assert len(iq) == expected

    def test_signal_non_negative_real_part(self):
        iq, _ = self.enc.encode(0)
        assert np.all(iq.real >= 0.0)

    def test_known_signal_sum(self):
        custom = {TXParams.AMPLITUDE: [[1.0, 1.0, 1.0]]}
        enc = ADSBEncoder(tx_params_distributions=custom, seed=0)
        iq, params = enc.encode(0)
        assert params[TXParams.AMPLITUDE] == 1.0

        nonzero = np.count_nonzero(iq.real)
        preamble_pulses = 4
        data_pulses = 112
        assert nonzero == preamble_pulses + data_pulses

class TestConfigure:
    def setup_method(self):
        self.enc = ADSBEncoder(seed=42)

    def test_returns_none(self):
        result = self.enc.configure()
        assert result is None

    def test_updates_sample_rate(self):
        self.enc.configure(sample_rate=4e6)
        assert self.enc.sample_rate == 4e6

    def test_updates_tx_params(self):
        new_params = {
            TXParams.AMPLITUDE: [[0.5, 0.5, 1.0]],
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

    def test_sample_rate_affects_encode(self):
        self.enc.configure(sample_rate=4e6)
        iq, _ = self.enc.encode(0)
        expected = int(round(120.0 * 4e6 / 1e6))
        assert len(iq) == expected
