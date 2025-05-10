"""Constants used in the project."""
from __future__ import annotations

from enum import Enum

from river import drift

DRIFT_DETECTORS = {
    "JSWIN": drift.JSWIN,
    "ADWIN": drift.ADWIN,
    "KSWIN": drift.KSWIN,
    # "HDDM_A": drift.binary.HDDM_A,
    # "HDDM_W": drift.binary.HDDM_W,
    "PH": drift.PageHinkley,
}


class DriftType(Enum):
    """Drift types."""

    ERROR = 1
    CONCEPT = 2  # real drift or virtual drift
    NO_DRIFT = 3
