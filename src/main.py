import argparse

import numpy as np
from sklearn.ensemble import *
from sklearn.datasets import *
from sklearn.model_selection import ParameterSampler, KFold, train_test_split

from modAL.disagreement import *
from modAL.uncertainty import *
from modAL import ActiveLearner

from query_strategies import *
from param_spaces import *


def main(pred, params, n_iter, q_strat, built_in_data, data_name, test_ratio, train_ratio, labeled_ratio, splitter):
    pred_class = eval(pred)
    sampler = ParameterSampler(eval(params), n_iter)

    query_strategy = eval(q_strat)
    X = None
    y = None
    if built_in_data:
        # TODO: Data loading must be made better
        X, y = eval(data_name)(return_X_y=True)
    else:
        raise NotImplementedError("Haven't gotten around to this yet.")

    # TODO: Might lead to an error. Do that as X, y then
    X_train_and_val = None
    X_test = None
    y_train_and_val = None
    y_test = None

    if test_ratio:
        X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=test_ratio)
    # if no test set, use all of X and y for train_and_val
    else:
        X_train_and_val = X
        y_train_and_val = y

    train_and_val_n_splits = int(1 / (1 - train_ratio))
    train_and_val_splitter = eval(splitter)(n_splits=train_and_val_n_splits)

    # dictionary that keeps track of avg. validation score per model
    val_scores = {}

    for hyper_params in sampler:
        # TODO: Think of how to call val_scores's key according to hyper_param
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
            # TODO: Check that labeled and pool are correctly done

            # Define active learner
            predictor = pred_class(**hyper_params)
            learner = ActiveLearner(predictor, query_strategy, X_training=X_training, y_training=y_training)

            # Feel free to use condition of your choice
            while len(X_pool) > 0:
                # TODO: So anpassen, dass man X_training auch reinpacken könnte
                # TODO: Man sollte angeben können, wie viele points man auf einmal extracten will.
                query_idx, _ = learner.query(X_pool)
                learner.teach(X_pool[query_idx], y_pool[query_idx])
                # Move pool data to labeled data
                X_training = np.concatenate((X_training, X_pool[query_idx]))
                y_training = np.concatenate((y_training, y_pool[query_idx]))

                # Delete explored X_pool data
                X_pool = np.delete(X_pool, query_idx, axis=0)
                y_pool = np.delete(y_pool, query_idx, axis=0)

            # track validation score
            val_score = learner.score(X_val, y_val)
            local_val_scores.append(val_score)

    # TODO: Potentially do stuff with validation scores and test dataset
    if test_ratio:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, help='An sklearn predictor e.g. RandomForestClassifier')
    parser.add_argument('--params', type=str, help='Parameter search space to sample from e.g. RFC_TEST')
    parser.add_argument('--n_iter', type=int, help='Number of parameter settings that are produced.')
    parser.add_argument('--q_strat', type=str, help='A modAL query_strategy or from query_strategies.py')
    parser.add_argument('--built_in_data', type=bool,
                        help='Currently, always set True. True iff sklearn dataset is to be used', default='True')
    parser.add_argument('--data_name', type=str, help='sklearn dataset or path to a dataset')
    parser.add_argument('--test_ratio', type=float, help='What ratio to use for test set relative to total set')
    parser.add_argument('--train_ratio', type=float, help='What ratio to use for train set relative to validation set')
    parser.add_argument('--labeled_ratio', type=float,
                        help='What ratio to use as initial labeled set for active learning')
    parser.add_argument('--splitter', type=str, help='What sklearn Splitter Class to use', default='KFold')

    main(**vars(parser.parse_args()))
