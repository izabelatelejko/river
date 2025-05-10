"""Experiment class to evaluate drift detection methods."""
from __future__ import annotations

import random
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.const import DRIFT_DETECTORS, DriftType
from utils.drift_detectors_bundle import DriftDetectorsBundle
from utils.model_with_drift_detector import ModelWithDriftDetector
from utils.parameter_config import DriftDetectorsParamConfig, DriftDetectorsParamGrid


class Experiment:
    """Experiment to evaluate drift detection methods."""

    def __init__(
        self,
        model_instance: Any,
        data_stream: Any,
        window_size: int,
        drift_col_id: int,
        seed: int = 123,
        drift_detectors_params_grid: DriftDetectorsParamGrid = DriftDetectorsParamGrid(),
    ) -> None:
        """Initialize the experiment."""
        np.random.seed(seed)
        random.seed(seed)

        self.window_size = window_size
        self.model_instance = model_instance
        self.drift_col_id = drift_col_id
        self.data_stream = data_stream
        self.detectors_params_grid = drift_detectors_params_grid

        self.error_detectors_params = self._find_optimal_detectors_params(DriftType.ERROR)
        self.concept_detectors_params = self._find_optimal_detectors_params(DriftType.CONCEPT)

        self.error_detectors = DriftDetectorsBundle(
            window_size=window_size,
            model_instance=model_instance,
            drift_type=DriftType.ERROR,
            detectors_params=self.error_detectors_params,
        )
        self.concept_detectors = DriftDetectorsBundle(
            window_size=window_size,
            model_instance=model_instance,
            drift_type=DriftType.CONCEPT,
            detectors_params=self.concept_detectors_params,
        )
        self.no_drift = ModelWithDriftDetector(
            window_size, model_instance, DriftType.NO_DRIFT, None
        )

    def _find_optimal_detectors_params(self, drift_type: DriftType) -> DriftDetectorsParamConfig:
        """Find the optimal parameters for the drift detectors."""
        print(f"Finding optimal parameters for {drift_type.name.lower()} detectors")
        detectors_params = {}
        for drift_detector_name, drift_detector_instance in DRIFT_DETECTORS.items():
            params_acc = []
            for drift_detector_param in self.detectors_params_grid[drift_detector_name]:
                print(
                    f"Processing detector {drift_detector_name} with params {str(drift_detector_param)}"
                )
                curr_data_stream = self.data_stream.get_data_stream()
                model_with_detector = ModelWithDriftDetector(
                    self.window_size,
                    self.model_instance,
                    drift_type,
                    drift_detector_instance(**drift_detector_param.model_dump()),
                )
                for i, (x, y) in enumerate(curr_data_stream):
                    model_with_detector.run_iteration(i, x, y, self.drift_col_id)
                params_acc.append(model_with_detector.get_average_accuracy())

            best_param_id = np.argmax(np.array(params_acc))
            detectors_params[drift_detector_name] = self.detectors_params_grid[drift_detector_name][
                best_param_id
            ]

        return DriftDetectorsParamConfig(**detectors_params)

    def print_parameters(self) -> None:
        """Print the selected parameters for the drift detectors."""
        print(str(self.error_detectors_params))
        print(str(self.concept_detectors_params))

    def run(self) -> None:
        """Run the experiment on the data stream."""
        for i, (x, y) in enumerate(self.data_stream.get_data_stream()):
            for detector_name in DRIFT_DETECTORS.keys():
                self.error_detectors[detector_name].run_iteration(i, x, y, self.drift_col_id)
                self.concept_detectors[detector_name].run_iteration(i, x, y, self.drift_col_id)

            self.no_drift.run_iteration(i, x, y, self.drift_col_id)

    def get_average_accs(self) -> pd.DataFrame:
        """Get the average accuracies of the drift detectors."""
        df = pd.DataFrame([], columns=["drift_type", "avg_accuracy", "detector_name"])

        for detector_name in DRIFT_DETECTORS.keys():
            row = {
                "drift_type": DriftType.CONCEPT.name,
                "avg_accuracy": self.concept_detectors[detector_name].get_average_accuracy(),
                "detector_name": detector_name,
            }
            df = pd.concat([df, pd.DataFrame([row])])

        for detector_name in DRIFT_DETECTORS.keys():
            row = {
                "drift_type": DriftType.ERROR.name,
                "avg_accuracy": self.error_detectors[detector_name].get_average_accuracy(),
                "detector_name": detector_name,
            }
            df = pd.concat([df, pd.DataFrame([row])])

        row = {
            "drift_type": DriftType.NO_DRIFT.name,
            "avg_accuracy": self.no_drift.get_average_accuracy(),
            "detector_name": "No Detector",
        }
        df = pd.concat([df, pd.DataFrame([row])]).reset_index(drop=True)

        return df.pivot(index="drift_type", values="avg_accuracy", columns="detector_name")

    def plot(self, plot_only_JSWIN: bool = False) -> None:
        """Plot the results of the experiment."""
        if not plot_only_JSWIN:
            plt.scatter(
                [i for i in range(len(self.no_drift.accs))],
                self.no_drift.accs,
                s=1,
                color="skyblue",
            )
            plt.title("No Drift")
            plt.xlabel("Time Steps")
            plt.ylabel("Model Accuracy")
            plt.show()

        self._plot_detectors(
            self.error_detectors,
            "Error Drift Detection",
            "Model Accuracy",
            plot_only_JSWIN,
        )
        self._plot_detectors(
            self.concept_detectors,
            "Concept Drift Detection",
            "Model Accuracy",
            plot_only_JSWIN,
        )
        self._plot_detectors(
            self.concept_detectors,
            "Concept Drift Detection",
            f"Concept({self.drift_col_id})",
            plot_only_JSWIN,
            self.data_stream.X[0],
        )

    def _plot_detectors(
        self,
        detectors: Any,
        title: str,
        ylabel: str,
        plot_only_JSWIN: bool = False,
        vals: Optional[List[float]] = None,
    ) -> None:
        """Plot the results of drift detectors."""
        use_accs = True if vals is None else False
        xlabel = "Time Steps"
        detector_params = (
            self.concept_detectors_params
            if "concept" in title.lower()
            else self.error_detectors_params
        )

        if not plot_only_JSWIN:
            fig, ax = plt.subplots(2, 2, sharey=True)
            fig.set_figheight(10)
            fig.set_figwidth(15)

            for i, detector_name in enumerate(DRIFT_DETECTORS.keys()):
                if use_accs:
                    vals = detectors[detector_name].accs
                ax[i % 2, i // 2].scatter([i for i in range(len(vals))], vals, s=1, color="skyblue")
                first = True
                for drift_x in detectors[detector_name].drifts:
                    if first:
                        ax[i % 2, i // 2].axvline(
                            x=drift_x,
                            color=DriftDetectorsBundle.COLORS[detector_name],
                            linestyle="--",
                            label=detector_name,
                        )
                        first = False
                    ax[i % 2, i // 2].axvline(
                        x=drift_x,
                        linestyle="--",
                        color=DriftDetectorsBundle.COLORS[detector_name],
                    )
                ax[i % 2, i // 2].set_title(
                    f"{detector_name} ({str(detector_params[detector_name])})",
                    fontsize=12,
                )
                ax[i % 2, i // 2].set_xlabel(xlabel)
                ax[i % 2, i // 2].set_ylabel(ylabel)

            # fig.legend(loc="center right", fontsize=12, bbox_to_anchor=(1.10, 0.5))
            plt.suptitle(title, fontsize=16)
            fig.tight_layout()
            plt.show()
        else:
            if use_accs:
                vals = detectors["JSWIN"].accs
            plt.scatter([i for i in range(len(vals))], vals, s=1, color="skyblue")

            drift_x = detectors["JSWIN"].drifts
            try:
                plt.axvline(x=drift_x, color="red", linestyle="--", label="JSWIN")
                plt.title(title + " - JSWIN")
            except:
                plt.title(title + " - No Drift Detected")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()
