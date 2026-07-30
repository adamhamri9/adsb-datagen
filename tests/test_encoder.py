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
