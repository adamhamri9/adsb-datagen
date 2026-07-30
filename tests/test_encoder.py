import pytest

from src.adsb_generator.encoder import ADSBEncoder, TXParams


class TestTXParams:
    def test_has_amplitude(self):
        assert TXParams.AMPLITUDE.value == "amplitude"

    def test_has_threshold(self):
        assert TXParams.THRESHOLD.value == "threshold"

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
            TXParams.THRESHOLD: [
                [0.02, 0.10, 0.5],
                [0.10, 0.35, 0.3],
                [0.35, 0.60, 0.2],
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
        enc = ADSBEncoder(tx_params_distributions=custom)
        with pytest.raises(ValueError, match="Invalid tx param key"):
            enc._validate_distributions()

    def test_rejects_invalid_key_by_string(self):
        custom = {"bad_key": [[0.0, 1.0, 1.0]]}
        enc = ADSBEncoder(tx_params_distributions=custom)
        with pytest.raises(ValueError, match="Invalid tx param key"):
            enc._validate_distributions()

    def test_accepts_string_key_amplitude(self):
        custom = {"amplitude": [[0.0, 0.5, 0.5], [0.5, 1.0, 0.5]]}
        enc = ADSBEncoder(tx_params_distributions=custom)
        enc._validate_distributions()

    def test_accepts_string_key_threshold(self):
        custom = {"threshold": [[0.0, 0.5, 0.5], [0.5, 1.0, 0.5]]}
        enc = ADSBEncoder(tx_params_distributions=custom)
        enc._validate_distributions()

    def test_rejects_min_greater_than_max(self):
        custom = {
            TXParams.AMPLITUDE: [[0.5, 0.0, 1.0]],
        }
        enc = ADSBEncoder(tx_params_distributions=custom)
        with pytest.raises(ValueError, match="Invalid range"):
            enc._validate_distributions()

    def test_rejects_weights_summing_too_low(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 1.0, 0.3]],
        }
        enc = ADSBEncoder(tx_params_distributions=custom)
        with pytest.raises(ValueError, match="Sum of weights"):
            enc._validate_distributions()

    def test_rejects_weights_summing_too_high(self):
        custom = {
            TXParams.AMPLITUDE: [[0.0, 0.5, 0.6], [0.5, 1.0, 0.6]],
        }
        enc = ADSBEncoder(tx_params_distributions=custom)
        with pytest.raises(ValueError, match="Sum of weights"):
            enc._validate_distributions()

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

    def test_contains_all_expected_params(self):
        result = self.enc._sample_tx_params()
        assert TXParams.AMPLITUDE in result
        assert TXParams.THRESHOLD in result

    def test_values_are_floats(self):
        result = self.enc._sample_tx_params()
        for val in result.values():
            assert isinstance(val, float)

    def test_amplitude_in_expected_range(self):
        for _ in range(100):
            result = self.enc._sample_tx_params()
            assert 0.05 <= result[TXParams.AMPLITUDE] <= 1.0

    def test_threshold_in_expected_range(self):
        for _ in range(100):
            result = self.enc._sample_tx_params()
            assert 0.02 <= result[TXParams.THRESHOLD] <= 0.6

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

    def test_string_keys_produce_txparams_enum_keys(self):
        custom = {"amplitude": [[0.1, 0.2, 0.5], [0.2, 0.3, 0.5]],
                  "threshold": [[0.1, 0.2, 1.0]]}
        enc = ADSBEncoder(tx_params_distributions=custom, seed=42)
        result = enc._sample_tx_params()
        for key in result:
            assert isinstance(key, TXParams)
        assert TXParams.AMPLITUDE in result
        assert TXParams.THRESHOLD in result

    def test_values_respect_narrow_custom_range(self):
        custom = {
            TXParams.AMPLITUDE: [[0.75, 0.80, 1.0]],
            TXParams.THRESHOLD: [[0.05, 0.06, 1.0]],
        }
        enc = ADSBEncoder(tx_params_distributions=custom, seed=42)
        for _ in range(100):
            result = enc._sample_tx_params()
            assert 0.75 <= result[TXParams.AMPLITUDE] <= 0.80
            assert 0.05 <= result[TXParams.THRESHOLD] <= 0.06
