from sklearn.manifold import MDS
from util import Metrics
from hypo import generate_distances_matrix, visualisation_2d
import numpy as np
from scipy.spatial.distance import squareform, pdist

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

def cmds(distances_matrix, eps=5e-8):
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
