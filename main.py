import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS
from util import Metrics, print_percents, percents


def generate_distances_matrix(N_bounds, M_bounds, metric, p):
    N = np.random.randint(*N_bounds)
    M = np.random.randint(*M_bounds)
    if metric == "mahalanobis" and N <= M: N, M = M + 1, N
    points = np.random.randn(N, M)

    kwargs = {"metric": metric, "p": p} if metric == "minkowski" else {"metric": metric}
    return squareform(pdist(points, **kwargs)), N, M

def get_eigenvalues(distances_squared_matrix, eps): 
    eigenvalues = np.linalg.eigh(distances_squared_matrix)[0]
    filtered_eigenvalues = eigenvalues[np.abs(eigenvalues) >= eps]
    
    return eigenvalues, filtered_eigenvalues


def print_h0_err_log(metric, N, M, print_eig, filtered_eigenvalues):
    print(f"Metric: {metric}, N: {N}, M: {M}\n")
    if print_eig: print(f"Filtered Eghenvalues: {filtered_eigenvalues}\n\n")

def print_h0_results(h0_counts, tests_count, metrics):
    for i in range(len(h0_counts)): print_percents(h0_counts[i], tests_count, text=f"RESULT for metric {metrics[i]}: ", force=True)


def check_h0(filtered_eigenvalues):
    return np.sum(filtered_eigenvalues < 0) > np.sum(filtered_eigenvalues > 0) 


def h0_test(N_bounds, M_bounds, metric, eps, print_eig, p):
    """
    N - count of points
    M - count of coordinates
    """

    distances_matrix, N, M = generate_distances_matrix(N_bounds, M_bounds, metric, p)
    _, filtered_eigenvalues = get_eigenvalues(distances_matrix ** 2, eps)

    if check_h0(filtered_eigenvalues): return "Success!"
    else: print_h0_err_log(metric, N, M, print_eig, filtered_eigenvalues)

    return (N, M)

def h0_metric_test(tests_count, N_bounds, M_bounds, metric, eps, print_eig, p):
    errors = []
    h0_count = 0

    for _ in range(tests_count):
        result = h0_test(N_bounds, M_bounds, metric, eps, print_eig, p)

        if result == "Success!": h0_count += 1
        else: errors.append(result)

        print_percents(h0_count, tests_count, text=f"Metric {metric}: ", parts_count=20)

    return h0_count, errors

def h0_ultimate_test(tests_count=10_000, metrics=Metrics.Good.value, 
                       N_bounds=(5, 500), M_bounds=(5, 500), eps=1e-9, 
                       print_eig=False, p=3):

    errors = []
    h0_counts = []
    for metric in metrics:
        h0_count, metric_errors = h0_metric_test(tests_count, N_bounds, M_bounds, metric, eps, print_eig, p)

        h0_counts.append(h0_count)
        errors.append(metric_errors)
    
    print_h0_results(h0_counts, tests_count, metrics)

    return h0_counts, errors


def mds(distances_matrix, K=2, n_init=4, random_state=44, visual_msd=False, metric=''):
    """
    K - count of coordinates after mds
    """

    mds = MDS(n_components=K, dissimilarity='precomputed', n_init=n_init, random_state=random_state)
    new_points = mds.fit_transform(distances_matrix)

    return new_points


def mds_test(tests_count=1, metrics=Metrics.Good.value, 
             N_bounds=(5, 500), M_bounds=(5, 500),
             K=2, visual_mds=True, p=3):
    
    for metric in metrics:
        for _ in range(tests_count):
            new_points = mds(generate_distances_matrix(N_bounds, M_bounds, metric, p)[0], K=K)
            title = f"MDS with {metric} " + (f"p={p} " if metric == "minkowski" else "") + "metric"
            if K == 2 and visual_mds: visualisation_2d(new_points, title=title)


def visualisation_2d(points, title="Title", xlabel="X", ylabel="Y"):
    plt.scatter(points[:, 0], points[:, 1], s=8, c='purple', alpha=0.5)
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel) 
    plt.title(title)
    plt.grid(True)
    plt.show()


# mds_test(metrics=Metrics.Good.value)
# h0_ultimate_test(tests_count=100, metrics=Metrics.Mahalanobis.value)

arr = []
for i in range(10, 100): 
    tests = 1000
    result = h0_ultimate_test(tests_count=tests, metrics=['sqeuclidean'], N_bounds=(i, i + 1), M_bounds=(10, 11))[0][0]
    arr.append(percents(result, tests))
print(arr)
print(np.column_stack(([1 + (i / 10) for i in range(90)], arr)))
print(np.array(list(zip([1 + (i / 10) for i in range(90)], arr))))
visualisation_2d(np.array(list(zip([1 + (i / 10) for i in range(90)], arr))))

# это тоже пока нормально не написал
# h0_ultimate_test(tests_count=1000, metrics=['mahalanobis'], N_bounds=(200, 250), M_bounds=(1, 200))
# visualisation_2d(np.array(arr), metric='mahalanobis')