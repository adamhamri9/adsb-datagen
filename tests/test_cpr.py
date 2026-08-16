import pytest

from src.adsb_generator.algorithms import ADSBAlgorithms


class TestEncodeCPR:
    def test_returns_tuple_of_two_integers(self):
        lat, lon = 52.2572, 3.91937
        result = ADSBAlgorithms.encode_cpr(lat, lon, odd=False)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    @pytest.mark.parametrize("odd", [False, True])
    def test_output_values_are_17_bit_unsigned(self, odd):
        # 17-bit integer range: [0, 131071]
        lat_cpr, lon_cpr = ADSBAlgorithms.encode_cpr(45.0, 10.0, odd=odd)

        assert 0 <= lat_cpr <= 0x1FFFF
        assert 0 <= lon_cpr <= 0x1FFFF

    @pytest.mark.parametrize(
        "lat,lon,is_odd",
        [
            (0.0, 0.0, False),
            (0.0, 0.0, True),
            (52.2572, 3.91937, False),
            (52.2572, 3.91937, True),
            (-33.8688, 151.2093, False),
            (-33.8688, 151.2093, True),
        ],
    )
    def test_is_deterministic(self, lat, lon, is_odd):
        assert ADSBAlgorithms.encode_cpr(lat, lon, is_odd) == ADSBAlgorithms.encode_cpr(lat, lon, is_odd)

    def test_even_and_odd_frames_produce_different_encodings(self):
        lat, lon = 52.2572, 3.91937

        even_encoding = ADSBAlgorithms.encode_cpr(lat, lon, odd=False)
        odd_encoding = ADSBAlgorithms.encode_cpr(lat, lon, odd=True)

        assert even_encoding != odd_encoding

    @pytest.mark.parametrize(
        "lat,lon",
        [
            (90.0, 180.0),
            (-90.0, -180.0),
            (88.0, 0.0),  # High latitude polar region (NL = 1 zone)
            (0.0, 180.0),  # Equator / Anti-meridian boundary
            (0.0, -180.0),
        ],
    )
    def test_handles_extreme_coordinates(self, lat, lon):
        even_lat, even_lon = ADSBAlgorithms.encode_cpr(lat, lon, odd=False)
        odd_lat, odd_lon = ADSBAlgorithms.encode_cpr(lat, lon, odd=True)

        assert 0 <= even_lat <= 0x1FFFF
        assert 0 <= even_lon <= 0x1FFFF
        assert 0 <= odd_lat <= 0x1FFFF
        assert 0 <= odd_lon <= 0x1FFFF

    @pytest.mark.parametrize(
        "lat, lon, odd, expected_lat_cpr, expected_lon_cpr",
        [
            (0.0, 0.0, False, 0, 0),
            (0.0, 0.0, True, 0, 0),
            (52.2572, 3.91937, False, 93000, 68496),
            (52.2572, 3.91937, True, 73974, 68496),
            (30.0444, 31.2357, False, 970, 73059),
            (30.0444, 31.2357, True, 121103, 73059),
            (-34.6037, -58.3816, False, 30503, 3492),
            (-34.6037, -58.3816, True, 43101, 3492),
            (35.6762, 139.6503, False, 123998, 89188),
            (35.6762, 139.6503, True, 111009, 89188),
            (-33.9249, 18.4241, False, 45331, 19592),
            (-33.9249, 18.4241, True, 57683, 19592),
            (61.2181, -149.9003, False, 26610, 100155),
            (61.2181, -149.9003, True, 4321, 100155),
            (1.3521, 103.8198, False, 29537, 29940),
            (1.3521, 103.8198, True, 29045, 29940),
            (51.5074, -0.1278, False, 76620, 128839),
            (51.5074, -0.1278, True, 57867, 128839),
        ],
    )
    def test_encode_cpr(self, lat, lon, odd, expected_lat_cpr, expected_lon_cpr):
        lat_cpr, lon_cpr = ADSBAlgorithms.encode_cpr(lat, lon, odd)
        assert lat_cpr == expected_lat_cpr, f"lat: {lat_cpr} != {expected_lat_cpr}"
        assert lon_cpr == expected_lon_cpr, f"lon: {lon_cpr} != {expected_lon_cpr}"