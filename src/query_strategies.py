from inspect import getmembers, isfunction
import modAL
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
import json
import numpy as np
from scipy.spatial.distance import cdist


def _normalize(x):
    """

    :param x: np.ndarray to normalize
    :return:
    """
    return (x - x.min()) / (x.max() - x.min())

def _gu_uncertainty(rf, X_pool):
    """

    :param rf: sklearn random forest
    :param X_pool: dataset to sample from
    :return: normalized uncertainty from Gu, Zydek and Jin 2015
    """
    # class_probs must be of the kind that more is better e.g. probability
    class_probs = rf.predict_proba(X_pool)
    sorted_probs = np.sort(class_probs, axis=1)
    prob_diffs = sorted_probs[:, -1] - sorted_probs[:, -2]
    return _normalize(prob_diffs)


def _gu_density(X_pool, k):
    """

    :param rf: sklearn random forest
    :param X_pool: dataset to sample from
    :param k: number of nearest neighbors to consider
    :return: normalized density from Gu, Zydek and Jin 2015
    """

    # entry i,j == dist between i-th and j-th row vectors
    dist = cdist(X_pool, X_pool, metric='euclidean')
    sorted_dist = np.sort(dist, axis=1)

    # Leave out first col since that's just the distance to itself
    k_neigh = sorted_dist[:, 1:1+k]
    dens = np.mean(k_neigh, axis=1)
    return _normalize(dens)


def call_q_strat(classifier, X_pool, X_training, q_strat_name, q_strat_dict):
    q_strat = eval(q_strat_name)

    if q_strat in [entropy_sampling, margin_sampling, uncertainty_sampling]:
        return q_strat(classifier, X_pool, **q_strat_dict)
    # custom q_strat
    else:
        q_strat(X_pool, X_training, **q_strat_dict)