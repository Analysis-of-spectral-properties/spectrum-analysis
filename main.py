from hypo import get_eigenvalues, generate_points, plus_minus_test
from util import Metrics, metrics_list
from scipy.spatial.distance import squareform, pdist
import numpy as np
import matplotlib.pyplot as plt

for metric in metrics_list:
    if metric == 'minkowski':
        plus_minus_test(metric=metric, p = 8 * 0.5)