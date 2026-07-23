from enum import Enum

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
