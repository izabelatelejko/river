from enum import Enum
import numpy as np
from pydantic import BaseModel
from typing import List, Any

class DriftType(Enum):
    ERROR = 1
    CONCEPT = 2 # real drift or virtual drift
    NO_DRIFT = 3


class JSWINParams(BaseModel):

    alpha: float


class ADWINParams(BaseModel):

    delta: float


class KSWINParams(BaseModel):

    alpha: float


class HDDMAParams(BaseModel):
    drift_confidence: float


class HDDMWParams(BaseModel):
    drift_confidence: float


class PHParams(BaseModel):
    min_instances: int
    delta: float


class DriftDetectorsParamGrid(BaseModel):

    JSWIN: List[JSWINParams] = [JSWINParams(alpha=alpha) for alpha in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]]
    ADWIN: List[ADWINParams] = [ADWINParams(delta=delta) for delta in [0.0001, 0.001]]
    KSWIN: List[KSWINParams] = [KSWINParams(alpha=alpha) for alpha in [0.0001, 0.001]]
    HDDM_A: List[HDDMAParams] = [HDDMAParams(drift_confidence=drift_confidence) for drift_confidence in [1e-20, 1e-15]]
    HDDM_W: List[HDDMWParams] = [HDDMWParams(drift_confidence=drift_confidence) for drift_confidence in [1e-20, 1e-15]]
    PH: List[PHParams] = [PHParams(min_instances=ph_params[0], delta=ph_params[1]) for ph_params in [(100, 0.5), (100, 0.4)]]

    def __getitem__(self, item):
        """Get the drift detector by name."""
        if item == "JSWIN":
            return self.JSWIN
        elif item == "ADWIN":
            return self.ADWIN
        elif item == "KSWIN":
            return self.KSWIN
        elif item == "HDDM_A":
            return self.HDDM_A
        elif item == "HDDM_W":
            return self.HDDM_W
        elif item == "PH":
            return self.PH
        else:
            raise KeyError(f"Drift detection method {item} not found.")


class DriftDetectorsParamConfig(BaseModel):

    JSWIN: JSWINParams
    ADWIN: ADWINParams
    KSWIN: KSWINParams
    HDDM_A: HDDMAParams
    HDDM_W: HDDMWParams
    PH: PHParams
    
    def __getitem__(self, item):
        """Get the drift detector by name."""
        if item == "JSWIN":
            return self.JSWIN
        elif item == "ADWIN":
            return self.ADWIN
        elif item == "KSWIN":
            return self.KSWIN
        elif item == "HDDM_A":
            return self.HDDM_A
        elif item == "HDDM_W":
            return self.HDDM_W
        elif item == "PH":
            return self.PH
        else:
            raise KeyError(f"Drift detection method {item} not found.")
