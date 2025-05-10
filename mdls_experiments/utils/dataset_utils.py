"""Module with dataset utilities for generating and plotting data streams."""

from typing import List, Optional

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from river import stream, datasets


class DataStream(ABC):
    """Abstract class for data streams."""

    def __init__(self) -> None:
        """Initialize the data stream."""
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self._initialize_stream()

    def get_data_stream(self) -> stream.iter_pandas:
        """Get the data stream."""
        return stream.iter_pandas(self.X, self.y)

    @abstractmethod
    def _initialize_stream(self) -> None:
        """Initialize the data stream."""
        pass

    @abstractmethod
    def plot_stream(self) -> None:
        """Plot the data stream."""
        pass


class LabelShiftDataStream(DataStream):
    """Data stream with simulated label shift."""

    def __init__(self, n: int = 10000, ratios: List[float] = [0.2, 0.8]) -> None:
        """Initialize the LabelShiftDataStream."""
        self.n = n
        self.ratios = ratios
        super().__init__()

    def _initialize_stream(self) -> None:
        """Initialize the data stream."""
        for r in self.ratios:
            n1 = int(self.n * r) // len(self.ratios)
            n2 = self.n // len(self.ratios) - n1
            X_pos = np.random.normal(0, 1, n1)
            X_neg = np.random.normal(1, 1, n2)

            X_first = np.concatenate((X_pos[:n1], X_neg[:n2]), axis=0)
            y_first = np.concatenate((np.ones(n1), np.zeros(n2)), axis=0)
            X_second = np.concatenate((X_pos[n1:], X_neg[n2:]), axis=0)
            y_second = np.concatenate((np.ones(n2), np.zeros(n1)), axis=0)

            p = np.random.permutation(len(X_first))
            X_first, y_first = X_first[p], y_first[p]
            p = np.random.permutation(len(X_second))
            X_second, y_second = X_second[p], y_second[p]
            X_r = np.concatenate((X_first, X_second), axis=0)
            y_r = np.concatenate((y_first, y_second), axis=0)
            X_r = pd.DataFrame(X_r)
            y_r = pd.Series(y_r).astype("int")
            if self.X is None:
                self.X = X_r.copy()
                self.y = y_r.copy()
            else:
                self.X = pd.concat([self.X, X_r])
                self.y = pd.concat([self.y, y_r])

    def plot_stream(self) -> None:
        """Plot the data stream."""
        colormap = np.array(["red", "blue"])
        plt.scatter(
            [i for i in range(len(self.X[0]))], self.X[0], s=1, color=colormap[self.y]
        )
        plt.title("Virtual Drift (Label Shift)")
        plt.show()


class HyperplaneStream(DataStream):
    """Data stream based on Hyperplane generator."""

    def __init__(
        self,
        seed: int = 42,
        n_drift_features: int = 2,
        n_features: int = 2,
        mag_change: int = 0.3,
    ):
        """Initialize the HyperPlaneStream."""
        self.seed = seed
        self.n_drift_features = n_drift_features
        self.n_features = n_features
        self.mag_change = mag_change
        super().__init__()

    def _initialize_stream(self) -> None:
        """Initialize the data stream."""
        gen = datasets.synth.Hyperplane(
            seed=self.seed,
            n_drift_features=self.n_drift_features,
            n_features=self.n_features,
            mag_change=self.mag_change,
        )
        dataset = iter(gen.take(10000))

        X_list, y_list = [], []
        for _x, _y in dataset:
            X_list.append(_x)
            y_list.append(_y)
        self.X = pd.DataFrame(X_list)
        self.y = pd.Series(y_list)

    def plot_stream(self) -> None:
        """Plot the data stream."""
        colormap = np.array(["red", "blue"])
        fig, ax = plt.subplots(1, 2, figsize=[10, 5])
        ax[0].scatter(
            [i for i in range(len(self.X[0]))], self.X[0], s=1, color=colormap[self.y]
        )
        ax[0].set_title("Feature 0")
        ax[1].scatter(
            [i for i in range(len(self.X[1]))], self.X[1], s=1, color=colormap[self.y]
        )
        ax[1].set_title("Feature 1")
        plt.show()


class SyntheticDataStream(DataStream):
    """Data stream with synthetic data drawn from different distributions."""

    def __init__(
        self,
        distribution_types,
        pos_distribution_params,
        neg_distribution_params,
        samples_lens,
    ):
        """Initialize the SyntheticDataStream."""
        self.distribution_types = distribution_types
        self.pos_distribution_params = pos_distribution_params
        self.neg_distribution_params = neg_distribution_params
        self.samples_lens = samples_lens
        super().__init__()

    def _initialize_stream(self) -> None:
        """Initialize the data stream."""
        for id in range(len(self.distribution_types)):
            n = self.samples_lens[id] // 2
            X_pos = self.distribution_types[id](*self.pos_distribution_params[id], n)
            X_neg = self.distribution_types[id](*self.neg_distribution_params[id], n)
            y_id = np.concatenate((np.ones(n), np.zeros(n)), axis=0)

            X_id = np.concatenate((X_pos, X_neg), axis=0)
            p = np.random.permutation(len(X_id))

            X_id, y_id = X_id[p], y_id[p]
            X_id = pd.DataFrame(X_id)
            y_id = pd.Series(y_id).astype("int")

            if self.X is None:
                self.X, self.y = X_id.copy(), y_id.copy()
            else:
                self.X = pd.concat([self.X, X_id])
                self.y = pd.concat([self.y, y_id])

    def plot_stream(self) -> None:
        """Plot the data stream."""
        colormap = np.array(["red", "blue"])
        plt.scatter(
            [i for i in range(len(self.X[0]))], self.X[0], s=1, color=colormap[self.y]
        )
        plt.title("Artificial Dataset")
        plt.show()
