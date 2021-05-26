from inspect import getmembers, isfunction
import modAL
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
import json
import numpy as np


def _gu_uncertainty(rf, X_pool, n_iter):
    """

    :param rf: sklearn random forest
    :param X_pool:
    :return:
    """
    class_probs = rf.predict_proba(X_pool)
    sorted_probs = np.sort(class_probs, axis=1)
    prob_diffs = sorted_probs[:, -1] - sorted_probs[:, -2]

    return np.sort[prob_diffs][-n_iter:]

def call_q_strat(classifier, X_pool, X_training, q_strat_name, q_strat_dict):
    q_strat = eval(q_strat_name)

    if q_strat in [entropy_sampling, margin_sampling, uncertainty_sampling]:
        return q_strat(classifier, X_pool, **q_strat_dict)
    # custom q_strat
    else:
        q_strat(X_pool, X_training, **q_strat_dict)