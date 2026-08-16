import pytest

from src.adsb_generator.algorithms import ADSBAlgorithms

class TestEncodeGroundTrack:
    def test_valid_north(self):
        valid, code = ADSBAlgorithms.encode_ground_track(0.0)
        assert valid == 1
        assert code == 0

    def test_valid_east(self):
        valid, code = ADSBAlgorithms.encode_ground_track(90.0)
        assert valid == 1
        assert code == 32

    def test_valid_south(self):
        valid, code = ADSBAlgorithms.encode_ground_track(180.0)
        assert valid == 1
        assert code == 64

    def test_valid_west(self):
        valid, code = ADSBAlgorithms.encode_ground_track(270.0)
        assert valid == 1
        assert code == 96

    def test_full_rotation_wraps(self):
        _, code0 = ADSBAlgorithms.encode_ground_track(0.0)
        _, code360 = ADSBAlgorithms.encode_ground_track(360.0)
        assert code0 == code360

    def test_wrap_past_360(self):
        _, code = ADSBAlgorithms.encode_ground_track(450.0)
        _, code90 = ADSBAlgorithms.encode_ground_track(90.0)
        assert code == code90

    def test_negative_degrees_wrap(self):
        _, code = ADSBAlgorithms.encode_ground_track(-90.0)
        _, code270 = ADSBAlgorithms.encode_ground_track(270.0)
        assert code == code270

    def test_invalid_returns_zero(self):
        valid, code = ADSBAlgorithms.encode_ground_track(45.0, valid=False)
        assert valid == 0
        assert code == 0

    def test_none_returns_zero(self):
        valid, code = ADSBAlgorithms.encode_ground_track(None)
        assert valid == 0
        assert code == 0

    def test_none_with_valid_false(self):
        valid, code = ADSBAlgorithms.encode_ground_track(None, valid=False)
        assert valid == 0
        assert code == 0

    def test_code_range(self):
        for deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            _, code = ADSBAlgorithms.encode_ground_track(float(deg))
            assert 0 <= code <= 128

    @pytest.mark.parametrize(
        "degrees,expected",
        [
            (0.0, 0),
            (90.0, 32),
            (180.0, 64),
            (270.0, 96),
            (2.8125, 1),
            (357.1875, 127),
        ],
    )
    def test_known_values(self, degrees, expected):
        valid, code = ADSBAlgorithms.encode_ground_track(degrees)
        assert valid == 1
        assert code == expected