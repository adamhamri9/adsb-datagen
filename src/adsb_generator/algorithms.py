import math

class ADSBAlgorithms():

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

    _NL_TABLE = [
        (87, 59), (83, 58), (79, 57), (76, 56), (73, 55),
        (70, 54), (66, 53), (63, 52), (60, 51), (57, 50),
        (53, 49), (50, 48), (46, 47), (43, 46), (40, 45),
        (37, 44), (34, 43), (31, 42), (28, 41), (25, 40),
        (22, 39), (19, 38), (16, 37), (13, 36), (9, 35),
        (6, 34), (3, 33), (0, 32),
    ]

    @classmethod
    def _nl(cls, lat: float) -> int:
        if abs(lat) >= 87:
            return 1
        for lat_min, nl in cls._NL_TABLE:
            if abs(lat) >= lat_min:
                return nl
        return 32

    @classmethod
    def encode_cpr(cls, lat: float, lon: float, odd: bool) -> tuple[int, int]:
        lat = max(-90.0, min(90.0, lat))
        lon = max(-180.0, min(180.0, lon))

        NB = 17
        factor = 1 << NB

        nz_lat = 59 if odd else 60
        dlat = 360.0 / nz_lat

        zone_lat = math.floor(lat / dlat)
        rem_lat = lat - zone_lat * dlat
        yz = int(round((rem_lat / dlat) * factor)) & (factor - 1)

        rlat = dlat * (yz / factor + zone_lat)
        rlat = max(-90.0, min(90.0, rlat))

        nl = cls._nl(rlat)

        dlon = 360.0 / nl
        zone_lon = math.floor(lon / dlon)
        rem_lon = lon - zone_lon * dlon
        xz = int(round((rem_lon / dlon) * factor)) & (factor - 1)

        return yz, xz

    @staticmethod
    def encode_altitude(alt: int) -> int:
        alt_clamped = max(-1000, min(50100, alt))
        n = (alt_clamped + 1000) // 25

        top_bits = (n >> 4) & 0x7F
        bottom_bits = n & 0x0F

        return (top_bits << 5) | (1 << 4) | bottom_bits

    @staticmethod
    def encode_ground_track(degrees: float, valid: bool = True) -> tuple[int, int]:
        if not valid or degrees is None:
            return 0, 0

        degrees = degrees % 360.0
        trk_code = round(degrees * 128.0 / 360.0)

        return 1, trk_code