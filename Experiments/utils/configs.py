from enum import Enum
from pydantic import BaseModel

class DriftType(Enum):
    ERROR = 1
    CONCEPT = 2 # real drift or virtual drift
    NO_DRIFT = 3

class DriftDetectorsConfig(BaseModel):

    jswin_alpha: float = 0.5
    # itd....