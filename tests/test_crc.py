import pytest

from src.adsb_generator.algorithms import ADSBAlgorithms


class TestEncodeCRC:
    def test_returns_112_bit_packet(self):
        bits = (1 << 88) - 1

        packet = ADSBAlgorithms.calculate_crc(bits)

        assert packet.bit_length() <= 112

    def test_original_message_is_preserved(self):
        bits = 0x123456789ABCDEF123456789ABCD

        packet = ADSBAlgorithms.calculate_crc(bits)

        assert packet >> 24 == bits

    def test_crc_is_24_bits(self):
        bits = 0xABCDEF1234567890123456789ABC

        packet = ADSBAlgorithms.calculate_crc(bits)

        crc = packet & 0xFFFFFF

        assert 0 <= crc <= 0xFFFFFF

    @pytest.mark.parametrize(
        "bits",
        [
            0,
            1,
            (1 << 111),
            (1 << 112) - 1,
            0xAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
            0x5555555555555555555555555555,
            0x123456789ABCDEF123456789ABCD,
        ],
    )
    def test_is_deterministic(self, bits):
        assert ADSBAlgorithms.calculate_crc(bits) == ADSBAlgorithms.calculate_crc(bits)

    @pytest.mark.parametrize(
        "bits",
        [
            0,
            1,
            (1 << 112) - 1,
            0x123456789ABCDEF123456789ABCD,
        ],
    )
    def test_crc_changes_when_message_changes(self, bits):
        modified = bits ^ 1

        crc1 = ADSBAlgorithms.calculate_crc(bits) & 0xFFFFFF
        crc2 = ADSBAlgorithms.calculate_crc(modified) & 0xFFFFFF

        assert crc1 != crc2


    @pytest.mark.parametrize(
        "bits,expected_crc",
        [
        (
            0x8D76CE88204C9072CB4820,
            0x9A504D,
        ),
        (
            0x8D7C7181215D01A0820820,
            0x4D8BF1,
        ),
        (
            0x8D7C7745226151A0820820,
            0x5CE9C2,
        ),
        (
            0x8D7C80AD2358F6B1E35C60,
            0xFF1925,
        ),
        (
            0x8D7C146525446074DF5820,
            0x738E90,
        ),
        (
            0x8C7C1474381DA443C6450A,
            0x369656,
        ),
        (
            0x8C7C451C423C52D692D953,
            0x855472,
        ),
        (
            0x8D89611348DB01C6EA41C4,
            0xC7B8BF,
        ),
        (
            0x8D7C1BE8581B66E9BD8CEE,
            0xDC1C9F,
        ),
        (
            0x8F7C629659A0A6F64D8BAA,
            0x09D3F0,
        ),
    ]
    )
    def test_known_vectors(self, bits, expected_crc):
        packet = ADSBAlgorithms.calculate_crc(int(bits))
        assert packet & 0xFFFFFF == expected_crc