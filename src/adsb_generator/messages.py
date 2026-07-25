import random
import string
from enum import Enum
from .adsb_math import ADSBMath

class ADSBMessageType(Enum):
    IDENTIFICATION = "identification"
    SURFACE_POSITION = "surface_position"
    AIRBORNE_POSITION = "airborne_position"
    AIRBORNE_VELOCITY = "airborne_velocity"

class ADSBMessage():
    def __init__(self, message_type_probs: dict = None):

        default_probs = {
            ADSBMessageType.IDENTIFICATION: 0.25,
            ADSBMessageType.SURFACE_POSITION: 0.25,
            ADSBMessageType.AIRBORNE_POSITION: 0.25,
            ADSBMessageType.AIRBORNE_VELOCITY: 0.25
        }

        self.message_type_probs = message_type_probs or default_probs
        self._validate_probailities()

        self._CALLSIGN_CHARSET = {
            " ": 0,
            **{chr(ord("A") + i): i + 1 for i in range(26)},
            **{str(i): 48 + i for i in range(10)},
        }


    def _validate_probailities(self):
        valid_types = set(item for item in ADSBMessageType)
        valid_values = set(item.value for item in ADSBMessageType)

        for msg_type in self.message_type_probs.keys():
            if msg_type not in valid_types and msg_type not in valid_values:
                raise ValueError(
                    f"Invalid message type key: '{msg_type}'. "
                    f"Valid options are the ADSBMessageType enums or: {valid_values}"
                )
            
        total= sum(self.message_type_probs.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Sum of probabilities must equal 1.0, got {total}"
            )

    def _build_identification_message(self) -> int:
        tc = random.randint(1, 4)
        ca = random.randint(0, 7)

        prefix = "".join(random.choices(string.ascii_uppercase, k=3))
        digits = str(random.randint(10, 99999))
        callsign = (prefix + digits).ljust(8, " ")[:8]

        callsign_48bit = 0
        for char in callsign:
            code = self._CALLSIGN_CHARSET.get(char, 0)
            callsign_48bit = (callsign_48bit << 6) | (code & 0x3F)

        return (tc << 51) | (ca << 48) | callsign_48bit

    def _build_surface_position_message(self) -> int:
        tc = random.randint(5, 8)
        mov = random.randint(0, 127)
        s, trk = ADSBMath.encode_ground_track(random.uniform(0, 360), random.choice([True, False]))
        t = random.choices([1, 0], [0.9, 0.1], k=1)[0]

        f = random.randint(0, 1)
        lat_cpr, lon_cpr = ADSBMath.encode_cpr(random.uniform(-90, 90), random.uniform(-180, 180), (f == 1))

        return ((tc << 51) | (mov << 44) | (s << 43) | (trk << 36) |
        (t << 35) | (f << 34) | (lat_cpr << 17) | lon_cpr)

    def _build_airborne_position_message(self) -> int:
        tc = random.choice([random.randint(9, 18), random.randint(20, 22)])
        ss = random.randint(0 ,3)
        saf = random.choice([0, 1])

        alt = ADSBMath.encode_altitude(random.randint(500, 45000))

        t = random.choices([1, 0], [0.9, 0.1], k=1)[0]

        f = random.randint(0, 1)
        lat_cpr, lon_cpr = ADSBMath.encode_cpr(random.uniform(-90, 90), random.uniform(-180, 180), (f == 1))

        return ((tc << 51) | (ss << 49) | (saf << 48) | (alt << 36) |
                (t << 35) | (f << 34) | (lat_cpr << 17) | lon_cpr)

    def _build_airborne_velocity_message(self) -> int:
        tc = 19
        st = random.randint(1, 4)
        ic = random.choice([0, 1])
        ifr = random.choice([0, 1])
        nucv = random.randint(0, 4)

        subtype = 0
        if st in [1, 2]:
            dew = random.choice([0, 1])
            vew = (random.randint(0, 1021) + 1) if st == 1 else (int(random.choice([x for x in range(0, 4085, 4)]) / 4) + 1)
            dns = random.choice([0, 1])
            vns = (random.randint(0, 1021) + 1) if st == 1 else (int(random.choice([x for x in range(0, 4085, 4)]) / 4) + 1)
            
            subtype = (dew << 21) | (vew << 11) | (dns << 10) | vns
        else:
            sh = random.choice([0, 1])
            hdg = round((random.uniform(0.0, 360.0) * 1024) / 360) % 1024
            t = random.choice([0, 1])
            ais = (random.randint(0, 1021) + 1) if st == 3 else (int(random.choice([x for x in range(0, 4085, 4)]) / 4) + 1)

            subtype = (sh << 21) | (hdg << 11) | (t << 10) | ais

        vrsrc = random.choice([0, 1])
        svr = random.choice([0, 1])
        vr = int(random.choice([x for x in range(0, 32641, 64)]) / 64) + 1
        res = 0
        sdif = random.choice([0, 1])
        dalt = int(random.choice([x for x in range(0, 3151, 25)]) / 25) + 1

        return ((tc << 51) | (st << 48) | (ic << 47) | (ifr << 46) | (nucv << 43) |
                (subtype << 21) | (vrsrc << 20) | (svr << 19) | (vr << 10) | (res << 8) | (sdif << 7) | dalt)


    def build(self) -> int:
        types = list(self.message_type_probs.keys())
        weights = list(self.message_type_probs.values())
        selected_type = random.choices(types, weights, k=1)[0]

        df = 17                       
        ca = random.randint(0, 7)        
        icao = random.getrandbits(24)   
        me = 0  

        if selected_type == ADSBMessageType.IDENTIFICATION:
            me = self._build_identification_message()  
        elif selected_type == ADSBMessageType.SURFACE_POSITION:
            me = self._build_surface_position_message()
        elif selected_type == ADSBMessageType.AIRBORNE_POSITION:
            me = self._build_airborne_position_message()
        elif selected_type == ADSBMessageType.AIRBORNE_VELOCITY:
            me = self._build_airborne_velocity_message()

        data = (df << 83) | (ca << 80) | (icao << 56) | me

        return ADSBMath.calculate_crc(data)