"""Model with drift detector."""

from river import metrics, preprocessing

from typing import Any, Optional
from collections import deque

import numpy as np

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
        self.accs = []
        self.metric = metrics.ClassificationReport()
        self.model = preprocessing.StandardScaler() | self.model_instance()
        self.sliding_window = deque(maxlen=window_size)

    def get_average_accuracy(self):
        """Get the average accuracy of the model."""
        assert len(self.accs) > 0
        return np.mean(np.array(self.accs))

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
            self.metric.update(y, y_pred)
        self.accs.append(self.metric._accuracy.get())
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
