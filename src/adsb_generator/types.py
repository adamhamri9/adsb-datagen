from enum import Enum

class MissingPolicy(Enum):
    RAISE = "raise"
    IGNORE = "ignore"
    DEFAULTS = "defaults"
    CONSTANTS = "constant"