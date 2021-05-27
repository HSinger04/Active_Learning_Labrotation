import argparse
import json
import numpy as np
import scipy.sparse
from sklearn.ensemble import *
from sklearn.datasets import *
from sklearn.model_selection import ParameterSampler, KFold, train_test_split
from modAL import ActiveLearner

from query_strategies import call_q_strat


def train(sampler, train_and_val_splitter, X_train_and_val, y_train_and_val, q_strat_name, q_strat_dict, labeled_ratio, pred_class):
    # dictionary that keeps track of avg. validation score per model
    val_scores = {}

    for hyper_params in sampler:
        local_val_scores = []
        for train_ind, val_ind in train_and_val_splitter.split(X_train_and_val):

            # Define validation dataset
            X_val = X_train_and_val[val_ind]
            y_val = y_train_and_val[val_ind]

            X_total_train = X_train_and_val[train_ind]
            y_total_train = y_train_and_val[train_ind]
            # Define labeled and unlabeled training dataset
            X_training, X_pool, y_training, y_pool = train_test_split(X_total_train, y_total_train,
                                                                      train_size=labeled_ratio)

            # Define active learner
            predictor = pred_class(**hyper_params)
            learner = ActiveLearner(predictor, call_q_strat, X_training=X_training, y_training=y_training)

            # TODO: Feel free to use condition of your choice
            while X_pool.shape[0] > 0:
                query_idx, _ = learner.query(X_pool, X_training, q_strat_name, q_strat_dict)
                learner.teach(X_pool[query_idx], y_pool[query_idx])
                # Move queried pool data to labeled data
                if isinstance(X_training, np.ndarray):
                    X_training = np.concatenate((X_training, X_pool[query_idx]))
                elif isinstance(X_training, scipy.sparse.csr_matrix):
                    X_training = scipy.sparse.vstack([X_training, X_pool[query_idx]])
                else:
                    raise NotImplementedError("X_training neither ndarray nor csr_matrix")
                y_training = np.concatenate((y_training, y_pool[query_idx]))

                # Delete explored X_pool data
                if isinstance(X_training, np.ndarray):
                    X_pool = np.delete(X_pool, query_idx, axis=0)
                elif isinstance(X_training, scipy.sparse.csr_matrix):
                    mask = np.ones(X_pool.shape[0], dtype=bool)
                    mask[query_idx] = False
                    X_pool = X_pool[mask]
                else:
                    raise NotImplementedError("X_pool neither ndarray nor csr_matrix")

                y_pool = np.delete(y_pool, query_idx, axis=0)

            # track validation score
            score_val = learner.score(X_val, y_val)
            local_val_scores.append(score_val)

        val_scores[str(hyper_params)] = local_val_scores
    return val_scores


def main(pred, params, n_iter, q_strat_name, q_strat_dict_path, built_in_data, data_name, test_ratio, train_ratio, labeled_ratio, splitter):
    pred_class = eval(pred)

    # Initiate parameter sampler
    sampler = None
    with open(params) as p:
        params_dict = json.load(p)
        sampler = ParameterSampler(params_dict, n_iter)

    q_strat_dict = {}
    if q_strat_dict_path:
        with open(q_strat_dict_path) as q:
            q_strat_dict = json.load(q)

    # dataset place holders
    X = None
    y = None
    X_train_and_val = None
    X_test = None
    y_train_and_val = None
    y_test = None

    if built_in_data:
        data_loader = eval(data_name)
        if data_name in {"fetch_20newsgroups", "fetch_20newsgroups_vectorized", "fetch_lfw_pairs", "fetch_rcv1"}:
            X_train_and_val, y_train_and_val = data_loader(return_X_y=True, subset="train")
            X_test, y_test = data_loader(return_X_y=True, subset="test")
        else:
            X, y = data_loader(return_X_y=True)
    else:
        raise NotImplementedError("Haven't gotten around to this yet.")

    # Only split if we don't have a dataset with a test set already
    if X_test == None:
        # ...and we want to split the set in the first place
        if test_ratio:
            X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=test_ratio)
        # if no test set, use all of X and y for train_and_val
        else:
            X_train_and_val = X
            y_train_and_val = y

    train_and_val_n_splits = int(1 / (1 - train_ratio))
    train_and_val_splitter = eval(splitter)(n_splits=train_and_val_n_splits)

    val_scores = train(sampler, train_and_val_splitter, X_train_and_val, y_train_and_val, q_strat_name, q_strat_dict, labeled_ratio, pred_class)

    # TODO: Also return predictors maybe for testing?
    # TODO: Potentially do stuff with validation scores and test dataset
    # TODO: Also return configuration for q_strat
    if test_ratio:
        raise NotImplementedError("Implement final part with test dataset")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, help='An sklearn predictor e.g. RandomForestClassifier')
    parser.add_argument('--params', type=str, help='Path to parameter search space to sample from.')
    parser.add_argument('--n_iter', type=int, help='Number of parameter settings that are to be sampled. '
                                                   'If you want to do a full grid search, ignore this.',
                        default=99999999)
    parser.add_argument('--q_strat_name', type=str, help='A modAL query_strategy or from query_strategies.py')
    parser.add_argument('--q_strat_dict_path', type=str, help='Path to kwargs for query_strategy. '
                                                         'Leave out if you dont want to specify kwargs.', default="")
    parser.add_argument('--built_in_data', type=bool,
                        help='Currently, always set True. True iff sklearn dataset is to be used', default='True')
    parser.add_argument('--data_name', type=str, help='sklearn dataset or path to a dataset')
    parser.add_argument('--test_ratio', type=float, help='What ratio to use for test set relative to total set')
    parser.add_argument('--train_ratio', type=float, help='What ratio to use for train set relative to validation set')
    parser.add_argument('--labeled_ratio', type=float,
                        help='What ratio to use as initial labeled set for active learning')
    parser.add_argument('--splitter', type=str, help='What sklearn Splitter Class to use', default='KFold')

    main(**vars(parser.parse_args()))
