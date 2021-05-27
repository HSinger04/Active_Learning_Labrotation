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


def _sorted_pairwise_dist(x, y, metric='sqeuclidean'):
    # entry i,j == dist between x's i-th and y's j-th row vectors
    dist = cdist(x, y, metric=metric)
    sorted_dist = np.sort(dist, axis=1)
    return sorted_dist


def _gu_uncertainty(rf, X_pool):
    """
    Note that sklearn assigns probabilities rather than
    the number of votes to an example!

    :param rf: sklearn random forest
    :param X_pool: dataset to sample from
    :return: normalized uncertainty from Gu, Zydek and Jin 2015
    """
    # class_probs must be of the kind that more is better e.g. probability
    class_probs = rf.predict_proba(X_pool)
    sorted_probs = np.sort(class_probs, axis=1)
    unc = sorted_probs[:, -1] - sorted_probs[:, -2]
    return _normalize(unc)


def _gu_density(X_pool, k, metric="sqeuclidean"):
    """

    :param X_pool: dataset to sample from
    :param k: number of nearest neighbors to consider
    :param metric: what metric to use. defaults to sqeuclidean like in gu
    :return: normalized density from Gu, Zydek and Jin 2015
    """

    sorted_dist = _sorted_pairwise_dist(X_pool, X_pool, metric=metric)
    # Leave out first col since that's just the distance to itself
    k_neigh = sorted_dist[:, 1:1+k]
    den = np.mean(k_neigh, axis=1)
    return _normalize(den)


def _gu_diversity(X_pool, X_training, k=1, metric="sqeuclidean"):
    """

    :param X_pool: dataset to sample from
    :param X_training: labeled dataset
    :param k: number of nearest neighbors to consider
    :param metric: what metric to use. defaults to sqeuclidean like in gu
    :return: for default kwargs, normalized diversity from Gu, Zydek and Jin 2015
    """
    sorted_dist = _sorted_pairwise_dist(X_pool, X_training, metric=metric)
    k_neigh = sorted_dist[:, :k]
    div = np.mean(k_neigh, axis=1)
    return _normalize(div)


def gu_sampling(rf, X_pool, X_training, version, k_den, k_div=1,
                metric_den="sqeuclidean", metric_div="sqeuclidean", weights=[1, 1, 1], n_iter=1):
    """ Performs sampling based on Gu, Zydek and Jin 2015

    :param rf: sklearn random forest
    :param X_pool: dataset to sample from. Can be np.ndarray or csr_matrix
    :param X_training: labeled dataset. Can be np.ndarray or csr_matrix
    :param version: Dictates the selection function. Either "intuitive" or "gu"
    :param k_den: number of nearest neighbors to consider for density
    :param k_div: number of nearest neighbors to consider for diversity. default 1 for gu version
    :param metric_den: what metric to use for density. defaults to sqeuclidean like in gu
    :param metric_div: what metric to use for diversity. defaults to sqeuclidean like in gu
    :param weights: how to weight unc, den and div. defaults to [1, 1, 1]
    :param n_iter: how many samples to query. default to 1
    :return:
    """
    weights = np.array(weights)

    unc = _gu_uncertainty(rf, X_pool)
    den = _gu_density(X_pool, k_den, metric=metric_den)
    div = _gu_diversity(X_pool, X_training, k=k_div, metric=metric_div)

    part_array = None
    if version == "intuitive":
        part_array = np.array([-unc, -den, -div])
    elif version == "gu":
        part_array = np.array([unc, den, -div])
    else:
        raise ValueError("version is not intuitive or gu, but: " + version)

    s_array = np.sum(weights * part_array, axis=1)
    # pick n_iter best rated samples
    query_idx = np.argsort(s_array)[:n_iter]

    mask = np.ones(X_pool.shape[0], dtype=bool)
    mask[query_idx] = False
    query_inst = X_pool[mask]

    return query_idx, query_inst


def call_q_strat(classifier, X_pool, X_training, q_strat_name, q_strat_dict):
    q_strat = eval(q_strat_name)

    if q_strat in [entropy_sampling, margin_sampling, uncertainty_sampling]:
        return q_strat(classifier, X_pool, **q_strat_dict)
    # custom q_strat
    else:
        q_strat(X_pool, X_training, **q_strat_dict)