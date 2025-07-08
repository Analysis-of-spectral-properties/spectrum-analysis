import numpy as np
from scipy.spatial.distance import squareform, pdist
from util import metrics_list


def testing_hypotheses(tests_count=10_000, metrics=metrics_list, 
                       N_bounds=(2, 500), M_bounds=(1, 500), 
                       eps=1e-9):
    """
    N - count of points
    M - count of coordinates
    """

    for metric in metrics:
        h0_count = 0

        for _ in range(tests_count):
            N = np.random.randint(*N_bounds)
            M = np.random.randint(*M_bounds)
            points = np.random.randn(N, M)

            distances_matrix = squareform(pdist(points, metric=metric)) ** 2
            eigenvalues = np.linalg.eigh(distances_matrix)[0]


            filtered_eigenvalues = eigenvalues[np.abs(eigenvalues) >= eps]

            if np.sum(filtered_eigenvalues < 0) > np.sum(filtered_eigenvalues > 0): h0_count += 1
            else: print(f"Metric: {metric}, N: {N}, M: {M} \nFiltered Eghenvalues: {filtered_eigenvalues}\n\n")

            if h0_count % 1000 == 0: print(h0_count)
            if ((percentages := h0_count // tests_count) % 10 == 0): print(f"{percentages}%")


testing_hypotheses(metrics=['euclidean'], N_bounds=(3, 100))