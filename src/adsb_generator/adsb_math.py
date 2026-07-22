class ADSBMath():

    CRC_POLY = 0xFFF409

    @classmethod
    def calculate_crc(cls, data: int) -> int:

        packet = data << 24
        remainder = packet

        for bit in range(111, 23, -1):
            if (remainder >> bit) & 1:
                remainder ^= cls.CRC_POLY << (bit - 24)

        crc = remainder & 0xFFFFFF

        return packet | crc

    