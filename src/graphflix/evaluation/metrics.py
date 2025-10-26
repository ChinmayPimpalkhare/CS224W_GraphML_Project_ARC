import numpy as np

def recall_at_k(ranks, k=10):
    return np.mean((ranks <= k).astype(float))

def ndcg_at_k(ranks, k=10):
    gains = (ranks <= k) / np.log2(ranks + 1)
    return np.mean(gains)