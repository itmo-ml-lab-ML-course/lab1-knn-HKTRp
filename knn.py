import math
from typing import Callable

from pandas import DataFrame


def even_kernel(x):
    if abs(x) > 1:
        return 0
    return 0.5


class SomeKernel:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, x):
        return (1 - abs(x) ** self.a) ** self.b


def gaussian(x):
    return (1 / (math.sqrt(2 * math.pi))) * math.exp(-0.5 * (x ** 2))


def triangle(x):
    if abs(x) > 1:
        return 0
    return 1 - abs(x)


class MinkovskyMetrics:
    def __init__(self, p):
        self.p = p

    def __call__(self, p1, p2):
        p1, p2 = p1[1], p2[1]
        diffs = [abs(x1 - x2) for x1, x2 in zip(p1, p2)]
        s = sum(diffs)
        result = math.pow(s, 1 / self.p)
        return result


def cosine_unsimilarity(p1, p2):
    p1, p2 = p1[1], p2[1]
    similarity = sum([x1 * x2 for x1, x2 in zip(p1, p2)]) / (
            sum(map(lambda x: x ** 2, p1)) * sum(map(lambda x: x ** 2, p2)))
    return 1 - similarity


def chebyshev_metrics(p1, p2):
    p1, p2 = p1[1], p2[1]
    return max([abs(x1 - x2) for x1, x2 in zip(p1, p2)])


class KnnClassifier:

    # Window size either k must be specified
    def __init__(self, kernel: Callable[[float], float],
                 metrics: Callable,
                 window_size: float = None,
                 k: int = None):
        self.kernel = kernel
        self.metrics = metrics
        self.window_size = window_size
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _get_h(self, point: tuple, other_points: DataFrame):
        if self.window_size:
            return self.window_size
        # Naive O(nlogn) algorithm
        points_list = list(other_points.iterrows())
        points_list.sort(key=lambda x: self.metrics(point, x))
        return self.metrics(point, points_list[self.k])

    def predict_for_point(self, point, other_points, points_classes, apriori_weights=None):
        classes_scores = [0] * len(set(points_classes))  # 0 for every class we have
        if not apriori_weights:
            apriori_weights = [1] * len(other_points)
        h = self._get_h(point, other_points)
        for dataset_point, point_class, apriori_weight \
                in zip(other_points.iterrows(), points_classes, apriori_weights):
            classes_scores[point_class] += self.kernel(self.metrics(point, dataset_point) / h) * apriori_weight
        return classes_scores.index(max(classes_scores))

    def predict(self, points, apriori_weights=None):
        f = self.predict_for_point
        return [f(point, self.X, self.y, apriori_weights) for point in points.iterrows()]

    def get_lowess_weights(self):
        self.X = self.X.reset_index()
        y = list(self.y)
        return [
            self.kernel(y[i] - self.predict_for_point(
                x,
                self.X.drop([i]),
                y[:i] + y[i + 1:]))
            for i, x in enumerate(self.X.iterrows())]
