import collections
import itertools
import random
import typing
import math

from river.base import DriftDetector


class KLWIN(DriftDetector):
    r"""Kullback-Leibler Windowing method for concept drift detection.

    Parameters
    ----------
    alpha
        Probability for the test statistic of the Kullback-Leibler test. The alpha parameter is
        very sensitive, therefore should be set below 0.01.
    window_size
        Size of the sliding window.
    stat_size
        Size of the statistic window.
    window
        Already collected data to avoid cold start.
    seed
        Random seed for reproducibility.

    Notes
    -----
    KLWIN (Kullback-Leibler Windowing) is a concept change detection method based
    on the Kullback-Leibler (KL) statistical test. KL-test is a statistical test with
    no assumption of underlying data distribution. KLWIN can monitor data or performance
    distributions. Note that the detector accepts one dimensional input as array.

    KLWIN maintains a sliding window $\Psi$ of fixed size $n$ (window_size). The
    last $r$ (stat_size) samples of $\Psi$ are assumed to represent the last
    concept considered as $R$. From the first $n-r$ samples of $\Psi$,
    $r$ samples are uniformly drawn, representing an approximated last concept $W$.

    The KL-test is performed on the windows $R$ and $W$ of the same size. KL
     -test compares the distance of the empirical cumulative data distribution $dist(R,W)$.

    A concept drift is detected by KLWIN if:

    $$
    dist(R,W) > \alpha
    $$

    The difference in empirical data distributions between the windows $R$ and $W$ is too large
    since $R$ and $W$ come from the same distribution.
    Examples
    --------
    """

    def __init__(
        self,
        alpha: float = 0.1,
        window_size: int = 100,
        stat_size: int = 10,
        seed: int | None = None,
        window: typing.Iterable | None = None,
    ):
        super().__init__()
        # TODO: assrts for alpha, window_size, stat_size

        if window_size < 0:
            raise ValueError("window_size must be greater than 0.")

        if window_size < stat_size:
            raise ValueError("stat_size must be smaller than window_size.")

        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed

        self._reset()

        if window:
            self.window = collections.deque(window, maxlen=self.window_size)

    def _reset(self):
        super()._reset()
        self.p_value = 0
        self.n = 0
        self.window = collections.deque(maxlen=self.window_size)
        self._rng = random.Random(self.seed)

    def update(self, x: float) -> None:
        """Update the detector with a new sample.

        Parameters
        ----------
        x
            New data sample the sliding window should add.

        Returns
        -------
        self

        """
        if self.drift_detected:
            self._reset()

        self.n += 1

        self.window.append(x)
        if len(self.window) >= self.window_size:
            rnd_window = [
                self.window[r]
                for r in self._rng.sample(
                    range(self.window_size - self.stat_size), self.stat_size
                )
            ]
            most_recent = list(
                itertools.islice(
                    self.window, self.window_size - self.stat_size, self.window_size
                )
            )
            psi_value = self._kl_distance(most_recent, rnd_window) + self._kl_distance(
                rnd_window, most_recent
            )

            if psi_value > self.alpha:
                self.drift_detected = True
                self.window = collections.deque(most_recent, maxlen=self.window_size)
            else:
                self.drift_detected = False
        else:
            self.drift_detected = False

    def _kl_distance(self, p: typing.List[float], q: typing.List[float]) -> float:
        """Calculate the Kullback-Leibler divergence between two distributions.

        Parameters
        ----------
        p
            First distribution.
        q
            Second distribution.

        Returns
        -------
        float
            Kullback-Leibler divergence.

        """
        return sum(
            p[i] * (math.log(p[i] / q[i]) if q[i] > 0 else 0) for i in range(len(p))
        )
