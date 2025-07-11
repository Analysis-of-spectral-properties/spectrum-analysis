from enum import Enum


metrics_list = ['euclidean',
                'seuclidean',
                'chebyshev',

                'mahalanobis',
                
                'minkowski',

                'cityblock',
                'sqeuclidean',
                'braycurtis', 
                'canberra',
                'correlation',
                'cosine'
                ]

Metrics = Enum('Metrics', [('Good', metrics_list[:3]), 
                           ('Mahalanobis', [metrics_list[3]]),
                           ('Minkowski', [metrics_list[4]]), 
                           ('Bad', metrics_list[5:])])


def percents(part, all): return round((part / all) * 100, 2)

def print_percents(part, all, parts_count=10, text="", force=False): 
    if not force and (all // parts_count) == 0: return
    if force or part % (all // parts_count) == 0: print(f"{text}{percents(part, all)}%\n")