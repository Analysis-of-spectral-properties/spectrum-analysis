import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS
from util import Metrics, print_percents


def testing_hypotheses(tests_count=10_000, metrics=Metrics.Euclidean.value, 
                       N_bounds=(2, 500), M_bounds=(1, 500), eps=1e-9,
                       visual_msd=False, make_mds=False, print_eig=False):
    """
    N - count of points
    M - count of coordinates
    """

    h0_counts = []
    for metric in metrics:
        h0_count = 0

        for _ in range(tests_count):
            N = np.random.randint(*N_bounds)
            M = np.random.randint(*M_bounds)
            points = np.random.randn(N, M)

            distances_matrix = squareform(pdist(points, metric=metric))
            distances_squared_matrix = distances_matrix ** 2
            eigenvalues = np.linalg.eigh(distances_squared_matrix)[0]

            if make_mds: mds(distances_matrix, visual_msd=visual_msd, metric=metric)

            filtered_eigenvalues = eigenvalues[np.abs(eigenvalues) >= eps]

            if np.sum(filtered_eigenvalues < 0) > np.sum(filtered_eigenvalues > 0): h0_count += 1
            else: 
                print(f"Metric: {metric}, N: {N}, M: {M} \n")
                if print_eig: print(f"Filtered Eghenvalues: {filtered_eigenvalues}\n\n")


            print_percents(h0_count, tests_count, text=f"Metric {metric}: ", parts_count=20)


        h0_counts.append(h0_count)
    
    
    for i in range(len(h0_counts)): print_percents(h0_counts[i], tests_count, text=f"RESULT for metric {metrics[i]}: ", force=True)
        


def mds(distances_matrix, K=2, n_init=4, random_state=44, visual_msd=False, metric=''):
    """
    K - count of coordinates after mds
    """

    mds = MDS(n_components=K, dissimilarity='precomputed', n_init=n_init, random_state=random_state)
    new_points = mds.fit_transform(distances_matrix)

    if K == 2 and visual_msd: visualisation_2d(new_points, metric)


def visualisation_2d(points, metric=''):
    plt.scatter(points[:, 0], points[:, 1])
    plt.title(f"MDS with {metric} metric")
    plt.show()


testing_hypotheses(tests_count=100, metrics=Metrics.Not_Euclidean.value, make_mds=False, visual_msd=False)