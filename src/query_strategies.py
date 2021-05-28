from inspect import getmembers, isfunction
import modAL
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
import json
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

# TODO: Normalize using some library e.g. sklearn
def _normalize(x):
    """

    :param x: np.ndarray to normalize
    :return:
    """
    return (x - x.min()) / (x.max() - x.min())


def _dist_batch_elemental(x_batch, y, y_batch_idx, start, end, metric):
    y_batch = None
    if end == None:
        y_batch = y[start:]
    else:
        y_batch = y[start:y_batch_idx[end]]
    # calculate pair-wise distances between the batches
    dist_batch = cdist(x_batch.A, y_batch.A, metric=metric)
    return dist_batch


def _x_dist_batch(x_batch, y, y_batch_idx, metric):

    # collects the distances from x_batch to all of y
    x_dist_batch = []

    for j, y_slice in enumerate(y_batch_idx[:-1]):
        x_dist_batch.append(_dist_batch_elemental(x_batch, y, y_batch_idx, y_slice, j + 1, metric))

    # collect the last part
    x_dist_batch.append(_dist_batch_elemental(x_batch, y, y_batch_idx, y_batch_idx[-1], None, metric))

    # concatenate to np array
    x_dist_batch = np.concatenate(x_dist_batch, axis=1)

    return x_dist_batch


def _sorted_pairwise_dist(x, y, metric, batch_size):
    """Calculates pair-wise distances between row vectors of x and y.

    :param x: nd.array or csr_matrix to compare with y
    :param y: nd.array or csr_matrix to compare with x
    :param metric: What metric to use for distance calculation
    :param batch_size: If x or y is a csr_matrix, specify how many entries can be loaded
                       into memory at once. Higher batch_size gives more speed, but might lead to OOM error
    :return: np.ndarray of pair-wise distances
    """
    # entry i,j == dist between x's i-th and y's j-th row vectors
    # transform x or y into dense representation, since otherwise, it's not supported
    if not x.__class__ == y.__class__:
        raise ValueError("x and y must be of same class")
    if batch_size < x.shape[1]:
        raise ValueError("data's feature dimensionality exceeds allowed batch size!")

    # variable for all pair-wise distances
    dist = []

    if x.__class__ == csr_matrix:
        # batch_step says how many row vectors can be processed at once
        batch_step = batch_size // x.shape[1]
        # see for loop how this is used
        x_batch_idx = np.arange(x.shape[0], step=batch_step)
        y_batch_idx = np.arange(y.shape[0], step=batch_step)

        # a couple things that should always hold true
        assert batch_step >= 1
        assert x_batch_idx.shape[0] >= 1
        assert y_batch_idx.shape[0] >= 1

        # Leave out last element of x_batch_slices, since that needs special handling
        for i, x_slice in enumerate(x_batch_idx[:-1]):
            # take one batch of at most batch_size elements
            x_batch = x[x_slice:x_batch_idx[i+1]]
            x_dist_batch = _x_dist_batch(x_batch, y, y_batch_idx, metric)
            # record x_dist_batch
            dist.append(x_dist_batch)

        # collect the last part
        x_batch = x[x_batch_idx[-1]:]
        x_dist_batch = _x_dist_batch(x_batch, y, y_batch_idx, metric)
        # record
        dist.append(x_dist_batch)
        # turn into np.ndarray
        dist = np.concatenate(dist, axis=0)

    # if x and y are np.ndarrays
    else:
        dist = cdist(x, y, metric=metric)
    sorted_dist = np.sort(dist, axis=1)
    return sorted_dist


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
    return _normalize(unc)


def _gu_density(X_pool, k, batch_size, metric="sqeuclidean"):
    """Calculates density from Gu, Zydek and Jin 2015

    :param X_pool: dataset to sample from
    :param k: number of nearest neighbors to consider
    :param metric: what metric to use. defaults to sqeuclidean like in gu
    :param batch_size: batch_size for _sorted_pairwise_dist
    :return: normalized density from Gu, Zydek and Jin 2015
    """

    sorted_dist = _sorted_pairwise_dist(X_pool, X_pool, metric, batch_size)
    # Leave out first col since that's just the distance to itself
    k_neigh = sorted_dist[:, 1:1+k]
    den = np.mean(k_neigh, axis=1)
    return _normalize(den)


def _gu_diversity(X_pool, X_training, batch_size, k=1, metric="sqeuclidean"):
    """Calculates diversity from Gu, Zydek and Jin 2015

    :param X_pool: dataset to sample from
    :param X_training: labeled dataset
    :param k: number of nearest neighbors to consider
    :param metric: what metric to use. defaults to sqeuclidean like in gu
    :param batch_size: batch_size for _sorted_pairwise_dist
    :return: for default kwargs, normalized diversity from Gu, Zydek and Jin 2015
    """
    sorted_dist = _sorted_pairwise_dist(X_pool, X_training, metric, batch_size)
    k_neigh = sorted_dist[:, :k]
    div = np.mean(k_neigh, axis=1)
    return _normalize(div)


def gu_sampling(rf, X_pool, X_training, version, k_den, k_div=1,
                metric_den="sqeuclidean", metric_div="sqeuclidean", batch_size=9999999, weights=[1, 1, 1], n_iter=1):
    """ Performs sampling based on Gu, Zydek and Jin 2015

    :param rf: sklearn random forest
    :param X_pool: dataset to sample from. Can be np.ndarray or csr_matrix
    :param X_training: labeled dataset. Can be np.ndarray or csr_matrix
    :param version: Dictates the selection function. Either "intuitive" or "gu"
    :param k_den: number of nearest neighbors to consider for density
    :param k_div: number of nearest neighbors to consider for diversity. default 1 for gu version
    :param metric_den: what metric to use for density. defaults to sqeuclidean like in gu
    :param metric_div: what metric to use for diversity. defaults to sqeuclidean like in gu
    :param batch_size: batch_size for _sorted_pairwise_dist
    :param weights: how to weight unc, den and div. defaults to [1, 1, 1]
    :param n_iter: how many samples to query. default to 1
    :return:
    """

    if not len(weights) == 3:
        raise ValueError("Weights must be an iterable consisting of three numbers")

    unc = _gu_uncertainty(rf, X_pool)
    den = _gu_density(X_pool, k_den, batch_size, metric=metric_den)
    div = _gu_diversity(X_pool, X_training, batch_size, k=k_div, metric=metric_div)

    part_array = None
    if version == "intuitive":
        weights = [-weights[0], -weights[1], -weights[2]]
    elif version == "gu":
        weights = [weights[0], weights[1], -weights[2]]
    else:
        raise ValueError("version is not intuitive or gu, but: " + version)

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