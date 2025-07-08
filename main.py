import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS
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

            distances_matrix = squareform(pdist(points, metric=metric))
            distances_squared_matrix = distances_matrix ** 2
            eigenvalues = np.linalg.eigh(distances_squared_matrix)[0]

            mds(distances_matrix)


            filtered_eigenvalues = eigenvalues[np.abs(eigenvalues) >= eps]

            if np.sum(filtered_eigenvalues < 0) > np.sum(filtered_eigenvalues > 0): h0_count += 1
            else: print(f"Metric: {metric}, N: {N}, M: {M} \nFiltered Eghenvalues: {filtered_eigenvalues}\n\n")

            if h0_count % 1000 == 0: print(h0_count)


def mds(distances_matrix, K=2, n_init=4, random_state=44):
    """
    K - count of coordinates after mds
    """

    mds = MDS(n_components=K, dissimilarity='precomputed', n_init=n_init, random_state=random_state)
    new_points = mds.fit_transform(distances_matrix)

    if K == 2: visualisation_2d(new_points)


def visualisation_2d(points):
    plt.scatter(points[:, 0], points[:, 1])
    plt.title('MDS')
    plt.show()


testing_hypotheses(metrics=['euclidean'], N_bounds=(99, 100), M_bounds=(99, 100), tests_count=1)