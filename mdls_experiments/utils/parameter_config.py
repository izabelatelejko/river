"""Parameter configuration for drift detectors."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel


class JSWINParams(BaseModel):
    """Parameters for JSWIN drift detector."""

    alpha: float


class ADWINParams(BaseModel):
    """Parameters for ADWIN drift detector."""

    delta: float


class KSWINParams(BaseModel):
    """Parameters for KSWIN drift detector."""

    alpha: float


class HDDMAParams(BaseModel):
    """Parameters for HDDM_A drift detector."""

    drift_confidence: float


class HDDMWParams(BaseModel):
    """Parameters for HDDM_W drift detector."""

    drift_confidence: float


class PHParams(BaseModel):
    """Parameters for Page-Hinkley drift detector."""

    min_instances: int
    delta: float


class DriftDetectorsParamGrid(BaseModel):
    """Grid of parameters to test for each drift detector."""

    JSWIN: List[JSWINParams] = [
        JSWINParams(alpha=alpha) for alpha in [0.4, 0.45, 0.5, 0.55, 0.6]
    ]
    ADWIN: List[ADWINParams] = [
        ADWINParams(delta=delta) for delta in [0.001, 0.002, 0.005, 0.01]
    ]
    KSWIN: List[KSWINParams] = [
        KSWINParams(alpha=alpha) for alpha in [0.0001, 0.001, 0.005, 0.01, 0.05]
    ]
    HDDM_A: List[HDDMAParams] = [
        HDDMAParams(drift_confidence=drift_confidence)
        for drift_confidence in [1e-20, 1e-15, 1e-10, 1e-5, 1e-3]
    ]
    HDDM_W: List[HDDMWParams] = [
        HDDMWParams(drift_confidence=drift_confidence)
        for drift_confidence in [1e-20, 1e-15, 1e-10, 1e-5, 1e-3]
    ]
    PH: List[PHParams] = [
        PHParams(min_instances=ph_params[0], delta=ph_params[1])
        for ph_params in [
            (100, 0.005),
            (30, 0.005),
            (100, 0.05),
            (30, 0.05),
            (100, 0.0001),
            (30, 0.0001),
        ]
    ]

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
    """Set of parameters for all drift detectors."""

    JSWIN: JSWINParams
    ADWIN: ADWINParams
    KSWIN: KSWINParams
    # HDDM_A: HDDMAParams
    # HDDM_W: HDDMWParams
    PH: PHParams

    def __getitem__(self, item):
        """Get the drift detector by name."""
        if item == "JSWIN":
            return self.JSWIN
        elif item == "ADWIN":
            return self.ADWIN
        elif item == "KSWIN":
            return self.KSWIN
        # elif item == "HDDM_A":
        #     return self.HDDM_A
        # elif item == "HDDM_W":
        #     return self.HDDM_W
        elif item == "PH":
            return self.PH
        else:
            raise KeyError(f"Drift detection method {item} not found.")
