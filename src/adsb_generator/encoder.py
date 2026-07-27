from enum import Enum

class TXParam(Enum):
    TXPower = "transpowder_power"

class ADSBEncoder:
    def __init__(self, tx_param_distributions: dict = None):
        pass

    def _validate_distributions(self):
        pass

    def encode(self):
        pass