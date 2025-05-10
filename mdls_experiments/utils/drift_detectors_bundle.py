"""Drift detectors bundle of different methods."""
from __future__ import annotations

from typing import Any, Optional

from river import drift
from utils.const import DriftType
from utils.model_with_drift_detector import ModelWithDriftDetector
from utils.parameter_config import DriftDetectorsParamConfig


class DriftDetectorsBundle:
    """
    Drift detection result for a specific method.
    """

    COLORS = {
        "JSWIN": "slategrey",
        "ADWIN": "red",
        "KSWIN": "green",
        "PH": "orange",
        # "HDDM_A": "blue",
        # "HDDM_W": "purple",
    }

    def __init__(
        self,
        window_size: int,
        model_instance: Any,
        drift_type: DriftType,
        detectors_params: Optional[DriftDetectorsParamConfig],
    ):
        """Initialize the drift detectors."""
        self.window_size = window_size
        self.model_instance = model_instance
        self.drift_type = drift_type
        self.detectors_params = detectors_params
        self.jswin = ModelWithDriftDetector(
            window_size,
            model_instance,
            drift_type,
            drift.JSWIN(alpha=self.detectors_params.JSWIN.alpha, seed=42),
        )
        self.adwin = ModelWithDriftDetector(
            window_size,
            model_instance,
            drift_type,
            drift.ADWIN(delta=self.detectors_params.ADWIN.delta),
        )
        self.kswin = ModelWithDriftDetector(
            window_size,
            model_instance,
            drift_type,
            drift.KSWIN(alpha=self.detectors_params.KSWIN.alpha, seed=42),
        )
        # self.hddm_a = ModelWithDriftDetector(
        #     window_size,
        #     model_instance,
        #     drift_type,
        #     drift.binary.HDDM_A(drift_confidence=self.detectors_params.HDDM_A.drift_confidence),
        # )
        # self.hddm_w = ModelWithDriftDetector(
        #     window_size,
        #     model_instance,
        #     drift_type,
        #     drift.binary.HDDM_W(drift_confidence=self.detectors_params.HDDM_W.drift_confidence),
        # )
        self.ph = ModelWithDriftDetector(
            window_size,
            model_instance,
            drift_type,
            drift.PageHinkley(
                min_instances=self.detectors_params.PH.min_instances,
                delta=self.detectors_params.PH.delta,
            ),
        )

    def __getitem__(self, item):
        """Get the drift detector by name."""
        if item == "JSWIN":
            return self.jswin
        elif item == "ADWIN":
            return self.adwin
        elif item == "KSWIN":
            return self.kswin
        # elif item == "HDDM_A":
        #     return self.hddm_a
        # elif item == "HDDM_W":
        #     return self.hddm_w
        elif item == "PH":
            return self.ph
        else:
            raise KeyError(f"Drift detection method {item} not found.")
