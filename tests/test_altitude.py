import pytest

from src.adsb_generator.algorithms import ADSBAlgorithms


class TestEncodeAltitude:
    def test_returns_an_integer(self):
        result = ADSBAlgorithms.encode_altitude(0)

        assert isinstance(result, int)

    def test_output_is_13_bit_unsigned(self):
        for alt in [-1000, 0, 11000, 50100]:
            result = ADSBAlgorithms.encode_altitude(alt)
            assert 0 <= result <= 0x1FFF

    def test_is_deterministic(self):
        for alt in [0, 1000, 11000, -500]:
            assert ADSBAlgorithms.encode_altitude(alt) == ADSBAlgorithms.encode_altitude(alt)

    def test_bit4_always_set(self):
        for alt in [-1000, 0, 1000, 11000, 50100]:
            result = ADSBAlgorithms.encode_altitude(alt)
            assert result & (1 << 4) != 0, f"bit 4 not set for alt={alt}"

    @pytest.mark.parametrize(
        "alt,expected",
        [
            (-1000, 16),
            (-1, 87),
            (0, 88),
            (100, 92),
            (500, 124),
            (1000, 176),
            (3000, 336),
            (5000, 496),
            (10000, 888),
            (11000, 976),
            (30000, 2488),
            (50000, 4088),
            (50100, 4092),
        ],
    )
    def test_encode_altitude(self, alt, expected):
        result = ADSBAlgorithms.encode_altitude(alt)
        assert result == expected, f"alt={alt}: {result} != {expected}"

    @pytest.mark.parametrize(
        "alt,expected",
        [
            (-1001, 16),
            (-2000, 16),
            (50101, 4092),
            (60000, 4092),
        ],
    )
    def test_clamps_out_of_range(self, alt, expected):
        result = ADSBAlgorithms.encode_altitude(alt)
        assert result == expected, f"alt={alt}: {result} != {expected}"

    def test_output_increases_with_altitude(self):
        alts = [-1000, 0, 1000, 5000, 11000, 30000, 50100]
        encoded = [ADSBAlgorithms.encode_altitude(a) for a in alts]
        assert encoded == sorted(encoded)

    def test_25_foot_resolution(self):
        base = ADSBAlgorithms.encode_altitude(0)
        same_bucket = ADSBAlgorithms.encode_altitude(24)
        next_bucket = ADSBAlgorithms.encode_altitude(25)
        assert base == same_bucket
        assert next_bucket > base
