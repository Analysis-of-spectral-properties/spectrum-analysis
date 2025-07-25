from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS
from util import Metrics, percents, print_h0_results
from tqdm import tqdm
# from seer import cmdscale


def generate_points(N, M):
    """
    Принимает количество точек *N*, 
    размерность пространства "M"
    
    Возвращает рандомные точки.
    """

    return np.random.uniform(low=-500, high=500, size=(N, M)) + \
            20 * np.random.standard_t(13, size =(N, M)) + \
            20 * (np.random.standard_gamma(4, size=(N, M)) - 4)


def generate_distances_matrix(N_bounds, M_bounds, metric, p=3):
    """
    Принимает границы количества точек *N_bounds*, 
    границы размерности пространства "M_bounds", 
    метрику *metric*
    
    Возвращает рандомные точки, количество точек, размерность, сами точки.
    """

    N = np.random.randint(*N_bounds)
    M = np.random.randint(*M_bounds)
    if metric == "mahalanobis" and N <= M: N, M = M + 1, N
    points = generate_points(N, M)

    kwargs = {"metric": metric, "p": p} if metric == "minkowski" else {"metric": metric}
    return squareform(pdist(points, **kwargs)), N, M, points


def get_eigenvalues(distances_squared_matrix, eps): 
    """
    Принимает матрицу расстояний и eps (для отсечения шумовых нулей)

    Возвращает собственные числа и отфильтрованные собственные числа (не нулевые)
    """

    eigenvalues = np.linalg.eigh(distances_squared_matrix)[0]
    filtered_eigenvalues = eigenvalues[np.abs(eigenvalues) >= eps]
    
    return eigenvalues, filtered_eigenvalues


def check_h0(filtered_eigenvalues):
    """Проверка гипотезы о том, что количество отрицательных сч больше количества положительных"""

    return np.sum(filtered_eigenvalues < 0) > np.sum(filtered_eigenvalues > 0) 


def h0_test(N_bounds, M_bounds, metric, eps, print_eig, p):
    """
    Генерирует точки, матрицу расстояний и проверяет гипотезу

    Возвращает "Success", если гипотеза подтвердилась, иначе пару кол-ва точек и размерности
    """

    distances_matrix, N, M, _ = generate_distances_matrix(N_bounds, M_bounds, metric, p)
    _, filtered_eigenvalues = get_eigenvalues(distances_matrix ** 2, eps)

    if check_h0(filtered_eigenvalues): return "Success!"
    # else: print_h0_err_log(metric, N, M, print_eig, filtered_eigenvalues)

    return (N, M)


def h0_metric_test(tests_count=10000, N_bounds=(5, 500), M_bounds=(5, 500), 
                   metric=Metrics.Good.value[0], eps=1e-9, print_eig=False, 
                   p=3, disable=False, ncols=90):
    """
    Проводит по метрике *tests_count* h0_test

    Возвращает кол-во тестов, когда гипотеза верна, 
    кортеж из кол-ва точек и размерности матрицы тех тестов, где гипотеза была неверна
    """

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
    """
    По массиву метрик проводит по каждой *tests_count* h0_test
    """

    errors = []
    h0_counts = []
    for metric in metrics:
        h0_count, metric_errors = h0_metric_test(tests_count, N_bounds, M_bounds, metric, eps, print_eig, p, disable)

        h0_counts.append(h0_count)
        errors.append(metric_errors)
    
    print_h0_results(h0_counts, tests_count, metrics)

    return h0_counts, errors


def mds(distances_matrix, K=2, n_init=4, random_state=44, visual_msd=False, metric=''):
    """
    По матрице расстояний производит MDS
    """

    mds = MDS(n_components=K, dissimilarity='precomputed', n_init=n_init, random_state=random_state)
    new_points = mds.fit_transform(distances_matrix)

    return new_points


def mds_ultimate_test(tests_count=1, metrics=Metrics.Good.value, 
             N_bounds=(5, 500), M_bounds=(5, 500),
             K=2, visual_mds=True, p=3):
    """
    По массиву метрик, для каждой делает *tests_count* MDS

    visual_mds - флаг для визуализации результата
    """
    
    for metric in metrics:
        for _ in range(tests_count):
            new_points = mds(generate_distances_matrix(N_bounds, M_bounds, metric, p)[0], K=K)
            title = f"MDS with {metric} " + (f"p={p} " if metric == "minkowski" else "") + "metric"
            if K == 2 and visual_mds: visualisation_2d(new_points, title=title)


def ratio_n_m_h0_test(tests_count=100, metric=Metrics.Bad.value[0],
                      N_bounds=(10, 100), M=10, step=2, s=32, mode="plot"):
    """
    С помощью этой функции можно посмотреть, как с увеличением отношения числа точек 
    к размерности меняется выполнимость гипотезы

    Пока работает наивно, можно задать границы кол-ва точек, шаг и фиксированную размерность
    """

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
    """
    Просто визуализация точек
    """

    if mode == "scatter": plt.scatter(points[:, 0], points[:, 1], s=s, c='purple', alpha=0.5)
    else: plt.plot(points[:, 0], points[:, 1], c='purple', alpha=0.5)
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel) 
    plt.title(title)
    plt.grid(True)
    plt.show()


def cmds(distances_matrix, eps=5e-8):
    """
    По матрице расстояний восстанавливает точки в пространстве с Евклидовой метрикой
    """

    D = distances_matrix ** 2
    N = D.shape[0]


    B = D
    for i in range(N):
        B[i, :] = D[i, :] - D[i, :].mean()
    for i in range(N):
        B[:, i] = B[:, i] - B[:, i].mean()
    B *= -0.5

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # !!!
    eps = N * 2.22e-16 * np.max(eigenvalues)

    eigenvalues_pos = eigenvalues[eigenvalues > eps]
    U_d = eigenvectors.T[eigenvalues > eps].T

    return U_d @ np.diag(np.sqrt(eigenvalues_pos))

def cmds_test(N_bounds=(5, 10), M_bot=3, tol=1e-12):
    """
    Тестирует точность и корректность работы cmds, пока не дописано
    """

    N = np.random.randint(*N_bounds)
    print("first")
    D, _, M, points = generate_distances_matrix(N_bounds=(N, N + 1), M_bounds=(M_bot, N - 1), metric="euclidean")
    print("second")
    points_prediction = cmds(D, eps=tol *  np.linalg.norm(D, 'fro'))

    print(f"Gen: {points.shape}, Rec: {points_prediction.shape}")
    print(points.shape == points_prediction.shape)
    # if (delta := np.sum(D - squareform(pdist(points_prediction, "euclidean")))) > 5e-6: print(delta)
    # np.sum(points - points_prediction), 

    rec = squareform(pdist(points_prediction, "euclidean"))
    return np.sum(D - rec), np.linalg.norm(D - rec, 'fro') / np.linalg.norm(D, 'fro'), N, M


# for i in range(1):
# for i in tqdm(range(1000)): 
    # cmds_test(N_bounds=(10, 1000), eps=2e-6)
    # print(cmds_test(N_bounds=(5, 10)))
    # print()


"""
Ниже находятся примеры применения функций для проверки гипотез
"""

# h0_ultimate_test(tests_count=100000, metrics=Metrics.Good.value[:1], disable=True, eps=1e-12, N_bounds=(1, 6), M_bounds=(1, 9))
# mds_ultimate_test(metrics=Metrics.Good.value)

# for metric in Metrics.Bad.value: ratio_n_m_h0_test(metric=metric, tests_count=1000, step=2)
# for metric in Metrics.Bad.value: n_m_errors_test(metric=metric)

# ratio_n_m_h0_test(metric="cosine", tests_count=1000)
# n_m_errors_test(metric="cosine")

