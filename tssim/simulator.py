from typing import Sequence, Union

import numpy as np

from tssim import features


class Simulator():
    """
    Simulate time series data by applying (random) transformations to prior values or
    index value.
    """

    def __init__(self, features: Sequence[features.Feature]):
        self.features = features

    def simulate(self, start: Union[float, Sequence[float]], num: int) -> np.ndarray:
        values = [start] if isinstance(start, float) else list(start)

        for i in range(1, num):
            last = values[i - 1]
            next = self._simulate_data_point(i, last, values)
            values.append(next)

        return np.array(values)

    def _simulate_data_point(self, index: int, last: float, history: Sequence[float]) -> float:
        value = last

        for feature in self.features:
            value = feature(index, value, history)

        return value
