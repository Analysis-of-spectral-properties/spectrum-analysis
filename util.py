from enum import Enum


metrics_list = ['euclidean',
                
                'braycurtis', 
                'canberra', 
                'chebyshev',
                'cityblock',
                'correlation',
                'cosine',
                'hamming',
                'minkowski',
                'seuclidean',
                'sqeuclidean'
                ]

Metrics = Enum('Metrics', [('Euclidean', metrics_list[:1]), ('Not_Euclidean', metrics_list[1:])])


def percents(part, all): return (part / all) * 100

def print_percents(part, all, parts_count=10, text="", force=False): 
    if force or part % (all // parts_count) == 0: print(f"{text}{percents(part, all)}%\n")