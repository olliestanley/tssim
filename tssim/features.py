from typing import Optional, Sequence

import numpy as np


class Feature():
    def __call__(self, index: int, x: float, history: Sequence[float]) -> float:
        return self.transform(index, x, history)

    def transform(self, index: int, x: float, history: Sequence[float]) -> float:
        raise NotImplementedError("Feature not implemented!")


class Polynomial(Feature):
    def __init__(self, coefficients: Sequence[float], intercept: float = 0.0):
        self.coefficients = coefficients
        self.intercept = intercept

    def transform(self, index: int, x: float, history: Sequence[float]) -> float:
        out = 0

        for i, coefficient in enumerate(self.coefficients):
            out += coefficient * (x ** (i + 1))

        return out + self.intercept


class Limit(Feature):
    def __init__(
        self, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None
    ):
        if lower_bound and upper_bound and lower_bound > upper_bound:
            raise ValueError("Lower bound must not be greater than upper bound")

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def transform(self, index: int, x: float, history: Sequence[float]) -> float:
        if self.lower_bound:
            x = max(x, self.lower_bound)
        if self.upper_bound:
            x = min(x, self.upper_bound)

        return x


class Growth(Feature):
    def __init__(self, growth: float):
        self.growth = growth

    def transform(self, index: int, x: float, history: Sequence[float]) -> float:
        return x * (1 + self.growth)


class LongRangeGrowth(Feature):
    def __init__(self, growth: float, dependency_periods: int, dependency_decay: float = 0.0):
        self.growth = growth
        self.dependency_periods = dependency_periods
        self.dependency_decay = dependency_decay

    def transform(self, index: int, x: float, history: Sequence[float]) -> float:
        dependency_periods = min(self.dependency_periods, len(history))
        relevant_history = history[-dependency_periods:]

        weights = list(reversed([
            (1 - self.dependency_decay) ** (i + 1)
            for i in range(dependency_periods)
        ]))

        return x + (self.growth * np.average(relevant_history, weights=weights))


class Noise(Feature):
    def __init__(self, std: float, seed: Optional[int] = None):
        self.std = std
        self.rng = (
            np.random.default_rng() if not seed
            else np.random.default_rng(seed=seed)
        )

    def transform(self, index: int, x: float, history: Sequence[float]) -> float:
        factor = self.rng.normal(1.0, self.std)
        return x * factor


class SimpleSeason(Feature):
    def __init__(
        self,
        length: int,
        frequency: int,
        offset: int,
        max_factor: float,
        min_factor: float,
        function: str = "sin",
    ):
        if length > frequency:
            raise ValueError("Season may not overlap itself!")

        self.length = length
        self.frequency = frequency
        self.offset = offset
        self.max_factor = max_factor
        self.min_factor = min_factor
        self.function = np.cos if function == "cos" else np.sin

    def transform(self, index: int, x: float, history: Sequence[float]) -> float:
        period_index = index % self.frequency

        if (period_index < self.offset) or (period_index >= self.offset + self.length):
            return x

        season_index = period_index - self.offset

        if index - (season_index + 1) >= 0:
            base = history[index - (season_index + 1)]
        else:
            base = x

        func_input = (season_index / (self.length - 1)) * 2 * np.pi
        func_output = (self.function(func_input) + 1) / 2
        seasonality_range = self.max_factor - self.min_factor
        factor = self.min_factor + (seasonality_range * func_output)
        return base * factor
