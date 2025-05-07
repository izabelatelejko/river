from typing import Any, List

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from river import stream, datasets


class DataStream(ABC):
    def __init__(self):
        self.X = None
        self.y = None
        self._data_stream = self.initialize_stream()
    
    @abstractmethod
    def initialize_stream(self):
        pass
    
    @abstractmethod
    def plot_stream(self):
        pass
        
    def data_stream(self):
        return self._data_stream
    

class LabelShiftDataStream(DataStream):

    def __init__(self, n: int = 10000, ratios: List[float] = [0.2, 0.8]):
        self.n = n
        self.ratios = ratios
        
        super().__init__()

    def initialize_stream(self):
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
            y_r = pd.Series(y_r).astype('int')
            if self.X is None:
                self.X = X_r.copy()
                self.y = y_r.copy()
            else:
                self.X = pd.concat([self.X, X_r])
                self.y = pd.concat([self.y, y_r])

        return stream.iter_pandas(self.X, self.y)

    def plot_stream(self):
        colormap = np.array(['red', 'blue'])
        plt.scatter([i for i in range(len(self.X[0]))], self.X[0], s=1, color=colormap[self.y])
        plt.title('Virtual Drift (Label Shift)')
        plt.show()


class HyperPlaneStream(DataStream):
    def __init__(
            self,
            seed: int = 42,
            n_drift_features: int = 2,
            n_features: int = 2,
            mag_change:int = 0.3
        ):
        self.seed = seed
        self.n_drift_features = n_drift_features
        self.n_features = n_features
        self.mag_change = mag_change
        
        super().__init__()
    
    def initialize_stream(self):
        gen = datasets.synth.Hyperplane(
            seed=self.seed,
            n_drift_features=self.n_drift_features,
            n_features=self.n_features,
            mag_change=self.mag_change
        )
        dataset = iter(gen.take(10000))

        self.X, self.y = [], []
        for _x, _y in dataset:
            self.X.append(_x)
            self.y.append(_y)
        self.X = pd.DataFrame(self.X)
        self.y = pd.Series(self.y)

        return stream.iter_pandas(self.X, self.y)
    
    def plot_stream(self):
        colormap = np.array(['red', 'blue'])
        plt.scatter([i for i in range(len(self.X[0]))], self.X[0], s=1, color=colormap[self.y])
        plt.show()
        plt.scatter([i for i in range(len(self.X[1]))], self.X[1], s=1, color=colormap[self.y])
        plt.show()

class ArtificialDataStream(DataStream):
    def __init__(self, distribution_types, pos_distribution_params, neg_distribution_params, samples_lens):
        self.distribution_types = distribution_types
        self.pos_distribution_params = pos_distribution_params
        self.neg_distribution_params = neg_distribution_params
        self.samples_lens = samples_lens
        
        super().__init__()

    def initialize_stream(self):
        for id in range(1, len(self.distribution_types)):
            n = self.samples_lens[id] // 2
            X_pos = self.distribution_types[id](*self.pos_distribution_params[id], n)
            X_neg = self.distribution_types[id](*self.neg_distribution_params[id], n)
            y_id = np.concatenate((np.ones(n), np.zeros(n)), axis=0)

            X_id = np.concatenate((X_pos, X_neg), axis=0)
            p = np.random.permutation(len(X_id))

            X_id, y_id = X_id[p], y_id[p]
            X_id = pd.DataFrame(X_id)
            y_id = pd.Series(y_id).astype('int')

            if self.X is None:
                self.X, self.y = X_id.copy(), y_id.copy() 
            else:
                self.X = pd.concat([self.X, X_id])
                self.y = pd.concat([self.y, y_id])

        return stream.iter_pandas(self.X, self.y)
    
    def plot_stream(self):
        return super().plot_stream()
