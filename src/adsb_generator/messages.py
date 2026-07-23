import random
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

    def _validate_probailities(self):
        total= sum(self.message_type_probs.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Sum of probabilities must equal 1.0, got {total}"
            )

    def build(self) -> int:
        types = list(self.message_type_probs.keys())
        weights = list(self.message_type_probs.values())
        selected_type = random.choices(types, weights, k=1)[0]

        df = 17                       
        ca = random.randint(0, 7)        
        icao = random.getrandbits(24)     

        if selected_type == ADSBMessageType.IDENTIFICATION:
            pass  
        elif selected_type == ADSBMessageType.SURFACE_POSITION:
            pass
        elif selected_type == ADSBMessageType.AIRBORNE_POSITION:
            pass
        elif selected_type == ADSBMessageType.AIRBORNE_VELOCITY:
            pass
        else:
            pass

        random_me = random.getrandbits(56) # random message for testing

        data = (df << 83) | (ca << 80) | (icao << 56) | random_me

        return ADSBMath().calculate_crc(data)      