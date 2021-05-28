import re
import argparse
import json
import numpy as np
import scipy.sparse
from sklearn.ensemble import *
from sklearn.datasets import *
from sklearn.model_selection import ParameterSampler, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import *
from modAL import ActiveLearner
from time import time

from query_strategies import call_q_strat

MODELS = "models"
VAL_SCORES = "val_scores"
DEF_SCORE = "def_score"

# default settings for main and args
DEF_N_ITER = 99999999
DEF_Q_STRAT_DICT_PATH = ""
DEF_BUILT_IN_DATA = True
DEF_METRICS_PATH = ""
DEF_SPLITTER = "KFold"
DEF_RESULT_DIR = "../results"


def _extr_str_wo_last_slash(string):
    """Extract string without last /"""
    indices = [m.start(0) for m in re.finditer("/", string)]
    if indices:
        string = string[indices[-1]+1:]
    string = re.sub(".json", "", string)
    return string


def _get_metrics(learner, X, y_true, metrics):
    """Get various metric scores for learner on X

    :param learner: the learner to evaluate
    :param X: dataset to evaluate metrics on
    :param y_true: true labels of X
    :param metrics: dict of metric functions that use y_true, y_pred as args
    :return: dict with scores
    """
    metrics_vals = {}
    metrics_vals[DEF_SCORE] = learner.score(X, y_true)
    y_pred = learner.predict(X)
    for name, metric in metrics.items():
        metrics_vals[name] = metric(y_true, y_pred)

    return metrics_vals


def _load_data(built_in_data, data_name, test_ratio):
    """

    :param built_in_data: True iff sklearn dataset is to be used
    :param data_name: sklearn dataset or path to a dataset
    :param test_ratio: What ratio to use for test set relative to total set
    """
    # dataset place holders
    X = None
    y = None
    X_train_and_val = None
    X_test = None
    y_train_and_val = None
    y_test = None

    if built_in_data:
        data_loader = eval(data_name)
        # datasets where one can specify a subset argument
        if data_name in {"fetch_20newsgroups", "fetch_20newsgroups_vectorized", "fetch_lfw_pairs", "fetch_rcv1"}:
            X_train_and_val, y_train_and_val = data_loader(return_X_y=True, subset="train")
            X_test, y_test = data_loader(return_X_y=True, subset="test")
        else:
            X, y = data_loader(return_X_y=True)
    else:
        raise NotImplementedError("No support for non sklearn dataset yet.")

    # Only split if we don't have a dataset with a test set already
    if X_test == None:
        # ...and we want to split the set in the first place
        if test_ratio:
            X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=test_ratio)
        # if no test set, use all of X and y for train_and_val
        else:
            X_train_and_val = X
            y_train_and_val = y

    return X_train_and_val, y_train_and_val, X_test, y_test


def _train(sampler, train_and_val_splitter, X_train_and_val, y_train_and_val, q_strat_name, q_strat_dict, labeled_ratio, pred_class, metrics):
    """Trains active learner sampled from sampler, iterating over datasets from train_and_val_splitter."""
    # dictionary that keeps track of avg. validation score per model
    to_track = {}

    for hyper_params in sampler:
        print("Start with hyper_params: " + str(hyper_params))
        start_time = time()

        # initialize local_val_scores with empty lists
        local_val_scores = {DEF_SCORE: []}
        for name in metrics.keys():
            local_val_scores[name] = []

        # initialize list for tracking models
        models = []

        query_times = []

        # each data split can only be iterated over once
        num_iter = str(len([elem for elem in train_and_val_splitter.split(X_train_and_val, y_train_and_val)]))
        for i, (train_ind, val_ind) in enumerate(train_and_val_splitter.split(X_train_and_val, y=y_train_and_val)):

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

            # for tracking query time
            local_query_times = []

            init_num_X_pool = X_pool.shape[0]
            # TODO: Feel free to use condition of your choice
            while X_pool.shape[0] > init_num_X_pool // 2:
                # query and track time
                start = time()
                query_idx, _ = learner.query(X_pool, X_training, q_strat_name, q_strat_dict)
                local_query_times.append(time() - start)
                # teach learner
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
                mask = np.ones(X_pool.shape[0], dtype=bool)
                mask[query_idx] = False
                X_pool = X_pool[mask]
                y_pool = np.delete(y_pool, query_idx, axis=0)

            # track time
            query_times.append(np.mean(local_query_times))

            # track validation scores
            metric_scores = _get_metrics(learner, X_val, y_val, metrics)
            for name, metric_score in metric_scores.items():
                local_val_scores[name].append(metric_score)

            # track learner
            models.append(learner)

            print("Complete split #" + str(i + 1) + " out of " + num_iter)

        # record val score for this hyper param setting
        to_track[str(hyper_params)] = {VAL_SCORES: local_val_scores,
                                       MODELS: models,
                                       "query_time": query_times}
        print("Training time: " + str(time() - start_time) + "\n")

    return to_track


def _test(tracked_info, X_test, y_test, metrics):
    """Returns tracked_info with metrics of best validated model tested on X_test, y_test"""
    for k, v in tracked_info.items():
        # test model with best DEF_SCORE
        best_mod_idx = np.argsort(v[VAL_SCORES][DEF_SCORE])[-1]
        test_model = v[MODELS][best_mod_idx]
        v["test_scores"] = _get_metrics(test_model, X_test, y_test, metrics)

    return tracked_info


def _save_tracked_info(tracked_info, result_dir, params, q_strat_dict_path, q_strat_dict, data_name):
    """Processes and saves tracked_info as .json in result_dir
    using a file name composed of params, q_strat_dict_path and data_name"""
    # remove unnecessary information
    for v in tracked_info.values():
        del v[MODELS]

    # remove potential slash
    if result_dir[-1] == "/":
        result_dir = result_dir[:-1]

    # record some information
    tracked_info["q_strat_config"] = q_strat_dict

    params_part = _extr_str_wo_last_slash(params)
    q_strat_part = _extr_str_wo_last_slash(q_strat_dict_path)
    data_part = _extr_str_wo_last_slash(data_name)

    AND = ", "

    # Save information
    with open(result_dir + "/" + params_part + AND + q_strat_part + AND + data_part + ".json", "w") as result_file:
        json.dump(tracked_info, result_file, indent="\t")


def main(pred, params, q_strat_name, data_name, test_ratio, train_ratio, labeled_ratio, built_in_data=DEF_BUILT_IN_DATA,
         n_iter=DEF_N_ITER, splitter=DEF_SPLITTER, metrics_path=DEF_METRICS_PATH,
         q_strat_dict_path=DEF_Q_STRAT_DICT_PATH, result_dir=DEF_RESULT_DIR):
    """Trains an active learner and saves some interesting information in a .json file

    :param pred: An sklearn predictor e.g. RandomForestClassifier
    :param params: Path to parameter search space to sample from.
    :param n_iter: Number of parameter settings that are to be sampled. If you want to do a full grid search, ignore this.
    :param q_strat_name: A modAL query_strategy or from query_strategies.py
    :param q_strat_dict_path: Path to kwargs for query_strategy. Leave out if you dont want to specify kwargs.
    :param built_in_data: Currently, should always be set True. True iff sklearn dataset is to be used
    :param data_name: sklearn dataset or path to a dataset
    :param metrics_path: Path to metrics settings. If left out, only learner.score will be tracked.
    :param test_ratio: What ratio to use for test set relative to total set
    :param train_ratio: What ratio to use for train set relative to validation set
    :param labeled_ratio: What ratio to use as initial labeled set for active learning
    :param splitter: What sklearn Splitter Class to use
    :param result_dir: Where to save results. If default, will save in ../results
    """
    pred_class = eval(pred)

    # Initiate parameter sampler
    sampler = None
    with open(params) as p:
        params_dict = json.load(p)
        sampler = ParameterSampler(params_dict, n_iter)

    # Get the kwargs for our q_strat
    q_strat_dict = {}
    if q_strat_dict_path:
        with open(q_strat_dict_path) as q:
            q_strat_dict = json.load(q)

    # load metrics dict
    metrics = {}
    if metrics_path:
        with open(metrics_path) as m:
            metrics = json.load(m)
    for k, v in metrics.items():
        metrics[k] = eval(v)

    # Load data
    X_train_and_val, y_train_and_val, X_test, y_test = _load_data(built_in_data, data_name, test_ratio)

    train_and_val_n_splits = int(1 / (1 - train_ratio))
    # Object that will do e.g. KFold-Cross-Validation for us
    train_and_val_splitter = eval(splitter)(n_splits=train_and_val_n_splits)

    tracked_info = _train(sampler, train_and_val_splitter, X_train_and_val, y_train_and_val, q_strat_name, q_strat_dict,
                          labeled_ratio, pred_class, metrics)

    # test if there should be a test
    if test_ratio:
        tracked_info = _test(tracked_info, X_test, y_test, metrics)

    # save important information
    _save_tracked_info(tracked_info, result_dir, params, q_strat_dict_path, q_strat_dict, data_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, help='An sklearn predictor e.g. RandomForestClassifier')
    parser.add_argument('--params', type=str, help='Path to parameter search space to sample from.')
    parser.add_argument('--n_iter', type=int, help='Number of parameter settings that are to be sampled. '
                                                   'If you want to do a full grid search, ignore this.',
                        default=DEF_N_ITER)
    parser.add_argument('--q_strat_name', type=str, help='A modAL query_strategy or from query_strategies.py')
    parser.add_argument('--q_strat_dict_path', type=str, help='Path to kwargs for query_strategy. '
                                                         'Leave out if you dont want to specify kwargs.', default=DEF_Q_STRAT_DICT_PATH)
    parser.add_argument('--built_in_data', type=bool,
                        help='Currently, should always be set True. True iff sklearn dataset is to be used', default=DEF_BUILT_IN_DATA)
    parser.add_argument('--data_name', type=str, help='sklearn dataset or path to a dataset.')
    parser.add_argument('--metrics_path', type=str, help='Path to metrics settings.' 
                                                         'If left out, only learner.score will be tracked.', default=DEF_METRICS_PATH)
    parser.add_argument('--test_ratio', type=float, help='What ratio to use for test set relative to total set')
    parser.add_argument('--train_ratio', type=float, help='What ratio to use for train set relative to validation set')
    parser.add_argument('--labeled_ratio', type=float,
                        help='What ratio to use as initial labeled set for active learning')
    parser.add_argument('--splitter', type=str, help='What sklearn Splitter Class to use', default=DEF_SPLITTER)
    parser.add_argument('--result_dir', type=str, help='Where to save results. If default, will save in ../results', default=DEF_RESULT_DIR)

    main(**vars(parser.parse_args()))
