from enum import Enum


class DataVersions(Enum):
    V3 = 0,  # ONE-AUDIO-ONE-EPISODE
    V4 = 1,
    Vsavee = 2,  # SAVEE dataset
    Vimprov = 3  # MSP-IMPROV Dataset
