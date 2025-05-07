from river import drift

from typing import Any
from collections import deque

from utils import DriftType, ModelWithDriftDetector

class DriftDetorsBundle():
    """
    Drift detection result for a specific method.
    """

    DRIFTS = ["JSWIN", "ADWIN", "KSWIN", "HDDM_A", "HDDM_W", "PH"]
    COLORS = {"JSWIN": "slategrey", "ADWIN": "red", "KSWIN": "green", "PH": "orange", "HDDM_A": "blue", "HDDM_W": "purple"}

    def __init__(self, window_size: int, model_instance: Any, drift_type: DriftType):
        """Initialize the drift detectors."""
        self.window_size = window_size
        self.model_instance = model_instance
        self.drift_type = drift_type
        self.jswin = ModelWithDriftDetector(window_size, model_instance, drift_type, drift.JSWIN(alpha=0.55, seed=42))
        self.adwin = ModelWithDriftDetector(window_size, model_instance, drift_type, drift.ADWIN(delta=0.001))
        self.kswin = ModelWithDriftDetector(window_size, model_instance, drift_type, drift.KSWIN(alpha=0.001, seed=42))
        self.hddm_a = ModelWithDriftDetector(window_size, model_instance, drift_type, drift.binary.HDDM_A(drift_confidence=10e-20))
        self.hddm_w = ModelWithDriftDetector(window_size, model_instance, drift_type, drift.binary.HDDM_W(drift_confidence=10e-20))
        self.ph = ModelWithDriftDetector(window_size, model_instance, drift_type, drift.PageHinkley(min_instances=100, delta=0.5))

    def __getitem__(self, item):
        """Get the drift detector by name."""
        if item == "JSWIN":
            return self.jswin
        elif item == "ADWIN":
            return self.adwin
        elif item == "KSWIN":
            return self.kswin
        elif item == "HDDM_A":
            return self.hddm_a
        elif item == "HDDM_W":
            return self.hddm_w
        elif item == "PH":
            return self.ph
        else:
            raise KeyError(f"Drift detection method {item} not found.")