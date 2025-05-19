"""Model with drift detector."""
from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np

from river import metrics, preprocessing
from utils.const import DriftType


class ModelWithDriftDetector:
    """Model with drift detector."""

    def __init__(
        self,
        window_size: int,
        model_instance: Any,
        drift_type: DriftType,
        detector: Optional[Any],
    ):
        """Initialize the pair of model and drift detector."""
        self.window_size = window_size
        self.model_instance = model_instance
        self.drift_type = drift_type
        self.detector = detector
        self.drifts = []
        self.window_accs = []
        self.proper_classifications = []
        self.metric = metrics.ClassificationReport()
        self.model = preprocessing.StandardScaler() | self.model_instance()
        self.sliding_window = deque(maxlen=window_size)

    def get_windowed_accuracy(self):
        """Get the accuracy of the model from last window."""
        proper_len = len(self.proper_classifications)
        proper_count = np.sum(
            np.array(
                self.proper_classifications[max(0, proper_len - self.window_size):]
            )
        )
        return proper_count / self.window_size
    
    def get_average_accuracy(self):
        return np.mean(np.array(self.window_accs))

    def run_iteration(self, i: int, x: Any, y: Any, drift_col_id: Any) -> None:
        """Run one iteration of the model with drift detection."""
        chosen_x = {0: x[drift_col_id]}
        y_pred = self.model.predict_one(x)
        self._test(x, y, y_pred)
        self.model.learn_one(x, y)
        self._update_drift(chosen_x, y, y_pred)
        self._check_drift(i, verbose=0)

    def _test(self, x: Any, y: int, y_pred: Optional[int]) -> None:
        """Test the model on the input data."""
        if y_pred is not None:
            if y == y_pred:
                self.proper_classifications.append(1)
            else:
                self.proper_classifications.append(0)
        self.window_accs.append(self.get_windowed_accuracy())
        self.sliding_window.append((x, y))

    def _update_drift(self, x: Any, y: int, y_pred: Optional[int] = None) -> None:
        """Update the drift detector with the new data."""
        if self.detector is None:
            return
        if self.drift_type == DriftType.ERROR:
            self.detector.update(int(y_pred != y if y_pred is not None else 0))
        elif self.drift_type == DriftType.CONCEPT:
            self.detector.update(x[0])

    def _check_drift(self, i: int, verbose: int = 0) -> None:
        """Check if drift is detected."""
        if self.detector is None:
            return
        if verbose == 1:
            print(f"Drift detected at index {i}")
        if self.detector.drift_detected:
            self.drifts.append(i)
            self.model = preprocessing.StandardScaler() | self.model_instance()
            for x_win, y_win in self.sliding_window:
                self.model.learn_one(x_win, y_win)
            self.sliding_window.clear()
