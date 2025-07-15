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
    if force or part % (all // parts_count) == 0: print(f"{text}{percents(part, all)}%")

def print_h0_results(h0_counts, tests_count, metrics):
    print()
    for i in range(len(h0_counts)): print_percents(h0_counts[i], tests_count, text=f"RESULT for metric {metrics[i]}: ", force=True)