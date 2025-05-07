from typing import Any, Optional, List

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    ModelWithDriftDetector, 
    DriftDetorsBundle, 
    DriftType
)


class Experiment():
    """
    Experiment to evaluate drift detection methods.
    """
    # TODO: plot other metrics

    def __init__(
            self, 
            model_instance: Any,
            data_stream: Any,
            window_size: int,
            drift_col_id: int,
            seed: int = 123
        ):

        np.random.seed(seed)
        random.seed(seed)
        
        self.window_size = window_size
        self.model_instance = model_instance

        self.drift_col_id = drift_col_id
        self.data_stream = data_stream

        self.error_detectors = DriftDetorsBundle(window_size=window_size, model_instance=model_instance, drift_type=DriftType.ERROR)
        self.concept_detectors = DriftDetorsBundle(window_size=window_size, model_instance=model_instance, drift_type=DriftType.CONCEPT)
        self.no_drift = ModelWithDriftDetector(window_size, model_instance, DriftType.NO_DRIFT, None)


    def run(self):
        """Run the experiment on the data stream."""
        for i, (x, y) in enumerate(self.data_stream):
            for detector_name in DriftDetorsBundle.DRIFTS:
                self.error_detectors[detector_name].run_iteration(i, x, y, self.drift_col_id)
                self.concept_detectors[detector_name].run_iteration(i, x, y, self.drift_col_id)

            self.no_drift.run_iteration(i, x, y, self.drift_col_id)

    def plot(self, x: Any, plot_only_JSWIN: bool = False):
        """Plot the results of the experiment."""
        if not plot_only_JSWIN:
            plt.scatter([i for i in range(len(self.no_drift.accs))], self.no_drift.accs, s=1, color="skyblue")
            plt.title('No Drift')
            plt.xlabel("Time Steps")
            plt.ylabel("Model Accuracy")
            plt.show()

        self._plot_detectors(self.error_detectors, 'Error Drift Detection', "Model Accuracy", plot_only_JSWIN)
        self._plot_detectors(self.concept_detectors, 'Concept Drift Detection', "Model Accuracy", plot_only_JSWIN)
        self._plot_detectors(self.concept_detectors, 'Concept Drift Detection', f"Concept({self.drift_col_id})", plot_only_JSWIN, x[0])

    def _plot_detectors(self, detectors: Any, title: str, ylabel: str, plot_only_JSWIN: bool = False, vals: Optional[List[float]] = None):
        """Plot the results of drift detectors."""

        use_accs = True if vals is None else False
        xlabel = "Time Steps"

        if not plot_only_JSWIN:
            fig, ax = plt.subplots(2, 3, sharey=True)
            fig.set_figheight(10)
            fig.set_figwidth(15)

            for i, detector_name in enumerate(DriftDetorsBundle.DRIFTS):
                if use_accs:
                    vals = detectors[detector_name].accs
                ax[i % 2, i // 2].scatter([i for i in range(len(vals))], vals, s=1, color="skyblue")
                first = True
                for drift_x in detectors[detector_name].drifts:
                    if first:
                        ax[i % 2, i // 2].axvline(x=drift_x, color=DriftDetorsBundle.COLORS[detector_name], linestyle='--', label=detector_name)
                        first = False
                    ax[i % 2, i // 2].axvline(x=drift_x, linestyle='--', color=DriftDetorsBundle.COLORS[detector_name])
                ax[i % 2, i // 2].set_title(detector_name, fontsize=12)
                ax[i % 2, i // 2].set_xlabel(xlabel)
                ax[i % 2, i // 2].set_ylabel(ylabel)

            fig.legend(loc = "center right", fontsize=12, bbox_to_anchor=(1.10, 0.5))
            plt.suptitle(title, fontsize=16)
            fig.tight_layout()
            plt.show()
        else:
            if use_accs:
                vals = detectors["JSWIN"].accs
            plt.scatter([i for i in range(len(vals))], vals, s=1, color="skyblue")

            drift_x = detectors["JSWIN"].drifts
            try:
                plt.axvline(x=drift_x, color="red", linestyle='--', label="JSWIN")
                plt.title(title + " - JSWIN")
            except:
                plt.title(title + " - No Drift Detected")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()
