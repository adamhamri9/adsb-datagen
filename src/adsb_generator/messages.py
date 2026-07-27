import random
import string
from enum import Enum
from .adsb_math import ADSBMath

class ADSBMessageType(Enum):
    """Supported Automatic Dependent Surveillance-Broadcast (ADS-B) message types."""
    IDENTIFICATION = "identification"
    SURFACE_POSITION = "surface_position"
    AIRBORNE_POSITION = "airborne_position"
    AIRBORNE_VELOCITY = "airborne_velocity"

class ADSBMessage():
    """
    Generates synthetic 112-bit ADS-B (Mode S Downlink Format 17) raw frames.

    This class handles the probabilistic generation and bit-level encoding of 
    various ADS-B message types—including aircraft identification, surface position, 
    airborne position, and airborne velocity—complete with 24-bit CRC parity generation.

    Attributes:
        message_type_probs (dict[ADSBMessageType | str, float]): Mapping of message 
            types to their selection probabilities.
        seed: int | None = None: Seed for the internal random number generator to ensure reproducible 
            message output.
    """
    def __init__(self, message_type_probs: dict = None, seed: int | None = None):
        """
        Initializes the instance with message type probabilities.

        Args:
            message_type_probs: A mapping of ADSBMessageType to their respective 
                emission probabilities. If None, defaults to an equal 25% distribution 
                across IDENTIFICATION, SURFACE_POSITION, AIRBORNE_POSITION, and 
                AIRBORNE_VELOCITY.
            seed: Seed for the internal random number generator to ensure reproducible 
                message output. If None, the generator is seeded from system randomness.

        Raises:
            ValueError: If `message_type_probs` contains invalid probability values 
                or does not sum to 1.0 (validated via `_validate_probabilities`).
        """

        default_probs = {
            ADSBMessageType.IDENTIFICATION: 0.25,
            ADSBMessageType.SURFACE_POSITION: 0.25,
            ADSBMessageType.AIRBORNE_POSITION: 0.25,
            ADSBMessageType.AIRBORNE_VELOCITY: 0.25
        }

        self.message_type_probs = message_type_probs or default_probs
        self._validate_probabilities()

        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._rng = random.Random(self._seed)

        self._CALLSIGN_CHARSET = {
            " ": 0,
            **{chr(ord("A") + i): i + 1 for i in range(26)},
            **{str(i): 48 + i for i in range(10)},
        }

    @property
    def seed(self) -> int | None:
        """Gets the seed value passed at initialization."""
        return self._seed


    def _validate_probabilities(self):
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
        tc = self._rng.randint(1, 4)
        ca = self._rng.randint(0, 7)

        prefix = "".join(self._rng.choices(string.ascii_uppercase, k=3))
        digits = str(self._rng.randint(10, 99999))
        callsign = (prefix + digits).ljust(8, " ")[:8]

        callsign_48bit = 0
        for char in callsign:
            code = self._CALLSIGN_CHARSET.get(char, 0)
            callsign_48bit = (callsign_48bit << 6) | (code & 0x3F)

        return (tc << 51) | (ca << 48) | callsign_48bit

    def _build_surface_position_message(self) -> int:
        tc = self._rng.randint(5, 8)
        mov = self._rng.randint(0, 127)
        s, trk = ADSBMath.encode_ground_track(self._rng.uniform(0, 360), self._rng.choice([True, False]))
        t = self._rng.choices([1, 0], [0.9, 0.1], k=1)[0]

        f = self._rng.randint(0, 1)
        lat_cpr, lon_cpr = ADSBMath.encode_cpr(self._rng.uniform(-90, 90), self._rng.uniform(-180, 180), (f == 1))

        return ((tc << 51) | (mov << 44) | (s << 43) | (trk << 36) |
        (t << 35) | (f << 34) | (lat_cpr << 17) | lon_cpr)

    def _build_airborne_position_message(self) -> int:
        tc = self._rng.choice([self._rng.randint(9, 18), self._rng.randint(20, 22)])
        ss = self._rng.randint(0 ,3)
        saf = self._rng.choice([0, 1])

        alt = ADSBMath.encode_altitude(self._rng.randint(500, 45000))

        t = self._rng.choices([1, 0], [0.9, 0.1], k=1)[0]

        f = self._rng.randint(0, 1)
        lat_cpr, lon_cpr = ADSBMath.encode_cpr(self._rng.uniform(-90, 90), self._rng.uniform(-180, 180), (f == 1))

        return ((tc << 51) | (ss << 49) | (saf << 48) | (alt << 36) |
                (t << 35) | (f << 34) | (lat_cpr << 17) | lon_cpr)

    def _build_airborne_velocity_message(self) -> int:
        tc = 19
        st = self._rng.randint(1, 4)
        ic = self._rng.choice([0, 1])
        ifr = self._rng.choice([0, 1])
        nucv = self._rng.randint(0, 4)

        subtype = 0
        if st in [1, 2]:
            dew = self._rng.choice([0, 1])
            vew = (self._rng.randint(0, 1021) + 1) if st == 1 else (int(self._rng.choice([x for x in range(0, 4085, 4)]) / 4) + 1)
            dns = self._rng.choice([0, 1])
            vns = (self._rng.randint(0, 1021) + 1) if st == 1 else (int(self._rng.choice([x for x in range(0, 4085, 4)]) / 4) + 1)
            
            subtype = (dew << 21) | (vew << 11) | (dns << 10) | vns
        else:
            sh = self._rng.choice([0, 1])
            hdg = round((self._rng.uniform(0.0, 360.0) * 1024) / 360) % 1024
            t = self._rng.choice([0, 1])
            ais = (self._rng.randint(0, 1021) + 1) if st == 3 else (int(self._rng.choice([x for x in range(0, 4085, 4)]) / 4) + 1)

            subtype = (sh << 21) | (hdg << 11) | (t << 10) | ais

        vrsrc = self._rng.choice([0, 1])
        svr = self._rng.choice([0, 1])
        vr = int(self._rng.choice([x for x in range(0, 32641, 64)]) / 64) + 1
        res = 0
        sdif = self._rng.choice([0, 1])
        dalt = int(self._rng.choice([x for x in range(0, 3151, 25)]) / 25) + 1

        return ((tc << 51) | (st << 48) | (ic << 47) | (ifr << 46) | (nucv << 43) |
                (subtype << 21) | (vrsrc << 20) | (svr << 19) | (vr << 10) | (res << 8) | (sdif << 7) | dalt)


    def build(self) -> int:
        """
        Generates a self._rng ADS-B message based on configured probabilities and calculates its CRC.

        self._rngly selects an ADS-B message type using the configured probability distribution, 
        constructs the message fields (Downlink Format, Capability, ICAO address, and Message payload), 
        and computes the final parity checksum.

        Returns:
            The complete 112-bit ADS-B message encoded as an integer, including 
            the 24-bit Parity/Interrogation ID (CRC).
        """
        types = list(self.message_type_probs.keys())
        weights = list(self.message_type_probs.values())
        selected_type = self._rng.choices(types, weights, k=1)[0]

        df = 17                       
        ca = self._rng.randint(0, 7)        
        icao = self._rng.getrandbits(24)   
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