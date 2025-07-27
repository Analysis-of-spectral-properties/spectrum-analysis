from util import Metrics, percents, print_h0_results
import numpy as np
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
from tqdm import tqdm
from decimal import Decimal

def generate_points(N, M):
    return np.random.uniform(low=-500, high=500, size=(N, M)) + \
            20 * np.random.standard_t(13, size =(N, M)) + \
            20 * (np.random.standard_gamma(4, size=(N, M)) - 4)

def generate_distances_matrix(N_bounds, M_bounds, metric, p=3):
    N = np.random.randint(*N_bounds)
    M = np.random.randint(*M_bounds)
    if metric == "mahalanobis" and N <= M: N, M = M + 1, N
    points = generate_points(N, M)

    kwargs = {"metric": metric, "p": p} if metric == "minkowski" else {"metric": metric}
    return squareform(pdist(points, **kwargs)), N, M, points


def get_eigenvalues(distances_squared_matrix, eps): 
    eigenvalues = np.linalg.eigh(distances_squared_matrix)[0]
    filtered_eigenvalues = eigenvalues[np.abs(eigenvalues) >= eps]
    
    return eigenvalues, filtered_eigenvalues


def check_h0(filtered_eigenvalues):
    return np.sum(filtered_eigenvalues < 0) > np.sum(filtered_eigenvalues > 0) 


def h0_test(N_bounds, M_bounds, metric, eps, print_eig, p):
    """
    N - count of points
    M - count of coordinates
    """

    distances_matrix, N, M, _ = generate_distances_matrix(N_bounds, M_bounds, metric, p)
    _, filtered_eigenvalues = get_eigenvalues(distances_matrix ** 2, eps)

    if check_h0(filtered_eigenvalues): return "Success!"
    # else: print_h0_err_log(metric, N, M, print_eig, filtered_eigenvalues)

    return (N, M)


def h0_metric_test(tests_count=10000, N_bounds=(5, 500), M_bounds=(5, 500), 
                   metric=Metrics.Good.value[0], eps=1e-9, print_eig=False, 
                   p=3, disable=False, ncols=90):
    errors = []
    h0_count = 0

    if not disable: print(f"\nMetric {metric}")
    progress = tqdm(total=tests_count, desc="h0 accepted", ncols=ncols, disable=disable)

    for _ in tqdm(range(tests_count), desc="Progress   ", ncols=ncols, disable=disable):
        result = h0_test(N_bounds, M_bounds, metric, eps, print_eig, p)

        if result == "Success!": 
            h0_count += 1
            progress.update(1)
        else: errors.append(result)

    progress.close()

    return h0_count, errors


def h0_ultimate_test(tests_count=10000, metrics=Metrics.Good.value, 
                       N_bounds=(5, 500), M_bounds=(5, 500), eps=1e-9, 
                       print_eig=False, p=3, disable=False):

    errors = []
    h0_counts = []
    for metric in metrics:
        h0_count, metric_errors = h0_metric_test(tests_count, N_bounds, M_bounds, metric, eps, print_eig, p, disable)

        h0_counts.append(h0_count)
        errors.append(metric_errors)
    
    print_h0_results(h0_counts, tests_count, metrics)

    return h0_counts, errors

def ratio_n_m_h0_test(tests_count=100, metric=Metrics.Bad.value[0],
                      N_bounds=(10, 100), M=10, step=2, s=32, mode="plot"):
    percents_list = []

    for i in tqdm(range(*N_bounds, step), desc="Progress", ncols=90): 
        h0_count = h0_metric_test(tests_count=tests_count, metric=metric, N_bounds=(i, i + 1), 
                                  M_bounds=(M, M + 1), disable=True)[0]
        percents_list.append(percents(h0_count, tests_count))
    
    visualisation_2d(np.array(list(zip([Decimal(i / M) for i in range(*N_bounds, step)], percents_list))), 
                     title=f"Зависимость частоты выполнения h0 от N/M в {metric} metric", 
                     xlabel="N/M", ylabel="h0, %", s=s, mode=mode)


def n_m_errors_test(tests_count=1000, metric=Metrics.Bad.value[0], N_bounds=(5, 500), M_bounds=(5, 500)):
    errors = h0_metric_test(tests_count=tests_count, metric=metric, N_bounds=N_bounds, M_bounds=M_bounds)[1]
    visualisation_2d(np.array(errors), title=f"Зависимость N от M при неверной h0 в {metric} metric",
                     xlabel="N", ylabel="M")


def visualisation_2d(points, title="Title", xlabel="X", ylabel="Y", s=16, mode="scatter"):
    if mode == "scatter": plt.scatter(points[:, 0], points[:, 1], s=s, c='purple', alpha=0.5)
    else: plt.plot(points[:, 0], points[:, 1], c='purple', alpha=0.5)
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel) 
    plt.title(title)
    plt.grid(True)
    plt.show()

def pos_eig(distances_squared_matrix):
    vals, f_vals = get_eigenvalues(distances_squared_matrix, 1e-9 * np.linalg.norm(distances_squared_matrix))
    print(vals)
    return sum(1 for x in f_vals if x > 0)

def neg_eig(distances_squared_matrix):
    vals, f_vals = get_eigenvalues(distances_squared_matrix, 1e-9 * np.linalg.norm(distances_squared_matrix))
    return sum(1 for x in f_vals if x < 0)

def plus_minus_test(number_of_obj_arr = [2, 5, 10, 20, 100, 200, 500, 1_000], dimension = 2, metric = Metrics.Good.value, p = 3.775):
    number_arr = [2, 5, 10, 20, 100, 200, 500, 1_000]
    pos_arr = []
    neg_arr = []
    for num_obj in number_arr:
        points = generate_points(num_obj, dimension)
        kwargs = {"metric": metric, "p": p} if metric == "minkowski" else {"metric": metric}
        distance_matrix = squareform(pdist(points, **kwargs))
        distance_squared_matrix = distance_matrix ** 2
        pos_arr.append(pos_eig(distance_squared_matrix))
        neg_arr.append(neg_eig(distance_squared_matrix))
        # plt.axvline(x=num_obj, color='gray', linestyle='--', alpha=0.5)

    plt.plot(number_arr, pos_arr, color = "green",
             marker = "o", linestyle = "--", label = "Positive eigenvalues",
             markersize = 3, linewidth = 1)
    plt.plot(number_arr, neg_arr, color = "red",
             marker = "o", linestyle = "--", label = f"Negative eigenvalues",
             markersize = 3, linewidth = 1)
    plt.grid()
    plt.xlabel("Number of points")
    plt.ylabel("Number of eigvalues")
    if p != 3.775:
        plt.title(f"Pos and neg eigs with {metric} metric and dim = {dimension} and p = {p}")
    else:
        plt.title(f"Pos and neg eigs with {metric} metric and dim = {dimension}")
    plt.legend()
    plt.show()

    return pos_arr, neg_arr