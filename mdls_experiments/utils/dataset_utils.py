"""Module with dataset utilities for generating and plotting data streams."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from sklearn.feature_extraction import FeatureHasher

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff

from river import datasets, stream


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

    def __init__(self, n: int = 10000, ratios: List[float] = [0.2, 0.8], seed: int = 100) -> None:
        """Initialize the LabelShiftDataStream."""
        self.n = n
        self.ratios = ratios
        np.random.seed(seed)
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
        plt.scatter([i for i in range(len(self.X[0]))], self.X[0], s=1, color=colormap[self.y])
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
        ax[0].scatter([i for i in range(len(self.X[0]))], self.X[0], s=1, color=colormap[self.y])
        ax[0].set_title("Feature 0")
        ax[1].scatter([i for i in range(len(self.X[1]))], self.X[1], s=1, color=colormap[self.y])
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
        seed: int = 100,
    ):
        """Initialize the SyntheticDataStream."""
        self.distribution_types = distribution_types
        self.pos_distribution_params = pos_distribution_params
        self.neg_distribution_params = neg_distribution_params
        self.samples_lens = samples_lens
        np.random.seed(seed)
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
        plt.scatter([i for i in range(len(self.X[0]))], self.X[0], s=1, color=colormap[self.y])
        plt.title("Artificial Dataset")
        plt.show()


class ElectricityDataStream(DataStream):
    """Data stream with synthetic data drawn from different distributions."""

    def __init__(self):
        """Initialize the SyntheticDataStream."""
        super().__init__()

    def _initialize_stream(self) -> None:
        """Initialize the data stream."""
        data, _ = arff.loadarff('utils/real_datasets/electricity.arff')
        electricity = pd.DataFrame(data).drop(["date", "day"], axis=1).iloc[:30000, :40000]

        electricity.loc[electricity["class"] == b'UP', "class"] = 1
        electricity.loc[electricity["class"] == b'DOWN', "class"] = 0

        self.X = electricity.drop("class", axis=1)
        self.y = electricity["class"]

    def plot_stream(self, feature) -> None:
        """Plot the data stream."""
        colormap = np.array(["red", "blue"])
        plt.scatter([i for i in range(self.X[[feature]].shape[0])], np.array(self.X[[feature]]), s=1, color=colormap[np.array(self.y, dtype=int)])
        plt.title("Electricity Dataset")
        plt.show()


class AirlinesDataStream(DataStream):
    """Data stream with synthetic data drawn from different distributions."""

    def __init__(self):
        """Initialize the SyntheticDataStream."""
        super().__init__()

    def _initialize_stream(self) -> None:
        """Initialize the data stream."""
        data, _ = arff.loadarff('utils/real_datasets/airlines.arff')
        airlines = pd.DataFrame(data).iloc[:10000, :]
        # print(airlines)
        airlines["Delay"] = np.array(airlines["Delay"]).astype(int)
        airlines["DayOfWeek"] = np.array(airlines["DayOfWeek"]).astype(int)

        n_features = 3
        hasher = FeatureHasher(n_features=n_features, input_type='string')

        columns_to_hash = ['AirportFrom', 'AirportTo', 'Airline']

        for column in columns_to_hash:
            hashed_column = hasher.transform([[val] for val in airlines[column]])
            hashed_df = pd.DataFrame(hashed_column.toarray(), columns=[f'{column.lower()}_hash_{i}' for i in range(n_features)])

            airlines = pd.concat([airlines.drop(column, axis=1), hashed_df], axis=1)

        self.X = airlines.drop("Delay", axis=1)
        self.y = airlines["Delay"]

    def plot_stream(self, feature) -> None:
        """Plot the data stream."""
        colormap = np.array(["red", "blue"])
        plt.scatter([i for i in range(self.X[[feature]].shape[0])], np.array(self.X[[feature]]), s=1, color=colormap[np.array(self.y, dtype=int)])
        plt.title("Airlines Dataset")
        plt.show()
