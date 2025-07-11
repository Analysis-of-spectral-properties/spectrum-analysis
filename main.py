from decimal import Decimal
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

def h0_metric_test(tests_count=10000, N_bounds=(5, 500), M_bounds=(5, 500), 
                   metric=Metrics.Good.value[0], eps=1e-9, 
                   print_eig=False, p=3):
    errors = []
    h0_count = 0

    for _ in range(tests_count):
        result = h0_test(N_bounds, M_bounds, metric, eps, print_eig, p)

        if result == "Success!": h0_count += 1
        else: errors.append(result)

        print_percents(h0_count, tests_count, text=f"Metric {metric}: ", parts_count=20)

    return h0_count, errors

def h0_ultimate_test(tests_count=10000, metrics=Metrics.Good.value, 
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


def mds_ultimate_test(tests_count=1, metrics=Metrics.Good.value, 
             N_bounds=(5, 500), M_bounds=(5, 500),
             K=2, visual_mds=True, p=3):
    
    for metric in metrics:
        for _ in range(tests_count):
            new_points = mds(generate_distances_matrix(N_bounds, M_bounds, metric, p)[0], K=K)
            title = f"MDS with {metric} " + (f"p={p} " if metric == "minkowski" else "") + "metric"
            if K == 2 and visual_mds: visualisation_2d(new_points, title=title)


def ratio_n_m_h0_test(tests_count=100, metric=Metrics.Bad.value[0],
                      N_bounds=(10, 100), M=10, step=5, s=32, mode="plot"):
    percents_list = []

    for i in range(*N_bounds, step): 
        h0_count = h0_ultimate_test(tests_count=tests_count, metrics=[metric], N_bounds=(i, i + 1), M_bounds=(M, M + 1))[0][0]
        percents_list.append(percents(h0_count, tests_count))
    
    visualisation_2d(np.array(list(zip([Decimal(i / M) for i in range(*N_bounds, step)], percents_list))), 
                     title=f"Зависимость частоты выполнения h0 от N/M в {metric} metric", 
                     xlabel="N/M", ylabel="h0, %", s=s, mode=mode)


def visualisation_2d(points, title="Title", xlabel="X", ylabel="Y", s=16, mode="scatter"):
    if mode == "scatter": plt.scatter(points[:, 0], points[:, 1], s=s, c='purple', alpha=0.5)
    else: plt.plot(points[:, 0], points[:, 1], c='purple', alpha=0.5)
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel) 
    plt.title(title)
    plt.grid(True)
    plt.show()


# это тоже пока нормально не написал
def n_m_errors_test(metric=Metrics.Bad.value[0], N_bounds=(5, 500), M_bounds=(5, 500)):
    errors = h0_ultimate_test(tests_count=1000, metrics=[metric], N_bounds=N_bounds, M_bounds=M_bounds)[1][0]
    visualisation_2d(np.array(errors), title=f"Зависимость N от M при неверной h0 в {metric} metric",
                     xlabel="N", ylabel="M")

# mds_ultimate_test(metrics=Metrics.Good.value)
# h0_ultimate_test(tests_count=100, metrics=Metrics.Mahalanobis.value)
# ratio_n_m_h0_test()
n_m_errors_test()