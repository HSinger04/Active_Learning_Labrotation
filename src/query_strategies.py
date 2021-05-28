from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import minmax_scale
from scipy.sparse import csr_matrix


def _pairwise_dist(x, y, metric):
    """Calculates pair-wise distances between row vectors of x and y.

    :param x: nd.array or csr_matrix to compare with y
    :param y: nd.array or csr_matrix to compare with x
    :param metric: What metric to use for distance calculation
    :return: np.ndarray of pair-wise distances
    """
    dist = None

    # needed as sqeuclidean implementation for sparse matrices doesnt exist
    if metric == "sqeuclidean" and (x.__class__ == csr_matrix or y.__class__ == csr_matrix):
        dist = pairwise_distances(x, y)
        dist = np.square(dist)
    else:
        try:
            dist = pairwise_distances(x, y, metric=metric)
        except TypeError as t:
            if x.__class__ == csr_matrix or y.__class__ == csr_matrix:
                raise TypeError("Specified metric does not support sparse matrices. "
                                "Check sklearn.metrics.pairwise_distance's documentation")
            else:
                raise t
    return dist


def _sorted_pairwise_dist(x, y, metric):
    """Calculates sorted pair-wise distances between row vectors of x and y.

    :param x: nd.array or csr_matrix to compare with y
    :param y: nd.array or csr_matrix to compare with x
    :param metric: What metric to use for distance calculation
    :return: sorted np.ndarray of pair-wise distances
    """

    dist = _pairwise_dist(x, y, metric)
    sorted_dist = np.sort(dist, axis=1)
    return sorted_dist


def _information_density(X_pool, k, metric):
    """Basically modAL.density.information_density, just a bit more general and normalized.

    :param X_pool: dataset to sample from
    :param k: dummy argument for common arguments with _gu_density
    :param metric: what metric to use.
    :return: sorted information density
    """
    dist = _pairwise_dist(X_pool, X_pool, metric)
    similarity_mtx = 1 / (1 + dist)
    similarities = similarity_mtx.mean(axis=1)
    return minmax_scale(similarities)


def _gu_uncertainty(rf, X_pool):
    """Calculates uncertainty from Gu, Zydek and Jin 2015

    :param rf: sklearn random forest
    :param X_pool: dataset to sample from
    :return: normalized uncertainty from Gu, Zydek and Jin 2015
    """
    # class_probs must be of the kind that more is better e.g. probability
    # HACK: Note that sklearn assigns probabilities rather than
    #     the number of votes to an example!
    class_probs = rf.predict_proba(X_pool)
    sorted_probs = np.sort(class_probs, axis=1)
    unc = sorted_probs[:, -1] - sorted_probs[:, -2]
    return minmax_scale(unc)


def _gu_density(X_pool, k, metric="sqeuclidean"):
    """Calculates density from Gu, Zydek and Jin 2015

    :param X_pool: dataset to sample from
    :param k: number of nearest neighbors to consider
    :param metric: what metric to use. defaults to sqeuclidean like in gu
    :return: normalized density from Gu, Zydek and Jin 2015
    """

    sorted_dist = _sorted_pairwise_dist(X_pool, X_pool, metric)
    # Leave out first col since that's just the distance to itself
    k_neigh = sorted_dist[:, 1:1+k]
    den = np.mean(k_neigh, axis=1)
    return minmax_scale(den)


def _gu_diversity(X_pool, X_training, k=1, metric="sqeuclidean"):
    """Calculates diversity from Gu, Zydek and Jin 2015

    :param X_pool: dataset to sample from
    :param X_training: labeled dataset
    :param k: number of nearest neighbors to consider
    :param metric: what metric to use. defaults to sqeuclidean like in gu
    :return: for default kwargs, normalized diversity from Gu, Zydek and Jin 2015
    """
    sorted_dist = _sorted_pairwise_dist(X_pool, X_training, metric)
    k_neigh = sorted_dist[:, :k]
    div = np.mean(k_neigh, axis=1)
    return minmax_scale(div)


def gu_sampling(rf, X_pool, X_training, k_den, k_div=1, metric_den="sqeuclidean", metric_div="sqeuclidean",
                weights=[1, 1, -1], n_iter=1, den_func="_gu_density"):
    """ Performs sampling based on Gu, Zydek and Jin 2015

    :param rf: sklearn random forest
    :param X_pool: dataset to sample from. Can be np.ndarray or csr_matrix
    :param X_training: labeled dataset. Can be np.ndarray or csr_matrix
    :param k_den: number of nearest neighbors to consider for density
    :param k_div: number of nearest neighbors to consider for diversity. default 1 for gu version
    :param metric_den: what metric to use for density. defaults to sqeuclidean like in gu
    :param metric_div: what metric to use for diversity. defaults to sqeuclidean like in gu
    :param weights: how to weight unc, den and div. defaults to gu's weights
    :param n_iter: how many samples to query. default to 1
    :param den_func: What density function to use. defaults to gu's
    :return:
    """

    if not len(weights) == 3:
        raise ValueError("Weights must be an iterable consisting of three numbers")

    unc = _gu_uncertainty(rf, X_pool)
    den = eval(den_func)(X_pool, k_den, metric=metric_den)
    div = _gu_diversity(X_pool, X_training, k=k_div, metric=metric_div)

    unc *= weights[0]
    den *= weights[1]
    div *= weights[2]

    s_array = unc + den + div
    # pick n_iter best rated samples
    query_idx = np.argsort(s_array)[:n_iter]

    mask = np.ones(X_pool.shape[0], dtype=bool)
    mask[query_idx] = False
    query_inst = X_pool[mask]

    return query_idx, query_inst


def call_q_strat(classifier, X_pool, X_training, q_strat_name, q_strat_dict):
    """Wrapper function for modAL and custom query strategies.

    :param classifier: the active learner's predictor
    :param X_pool: dataset to sample from. Can be np.ndarray or csr_matrix
    :param X_training: labeled dataset. Can be np.ndarray or csr_matrix
    :param q_strat_name: the name of the query_strategy
    :param q_strat_dict: additional kwargs for the query_strategy of interest
    :return: return value of query strategy of interest
    """
    q_strat = eval(q_strat_name)

    if q_strat in [entropy_sampling, margin_sampling, uncertainty_sampling]:
        return q_strat(classifier, X_pool, **q_strat_dict)
    # custom q_strat
    else:
        return q_strat(classifier, X_pool, X_training, **q_strat_dict)