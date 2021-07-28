import numpy as np
from numpy import dot
from numpy.linalg import norm


def cos_sim(A, B):
    return dot(A, B)/(norm(A) * norm(B))


def euclidean(A, B):
    distance = 0
    for idx, value in enumerate(A):
        distance += ((B[idx] - A[idx]) ** 2)
    distance = np.sqrt(distance)
    return distance


def min_max_normalization(list):
    return [
        (val - list.min()) /
        (list.max() - list.min())
        for val in
        list.values
    ]
