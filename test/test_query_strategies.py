import sys
from os.path import dirname, abspath
# Import src
sys.path.append(dirname(dirname(abspath(__file__))))
import numpy as np

from src import query_strategies

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, fetch_lfw_pairs


# Mock object for testing purposes
class Mock:
    def predict_proba(self, x):
        return x

def test__gu_uncertainty():
    # 3.2 - 1.4 == 1.8, 2.3 - 0.4 == 1.9, 1.0 - 0.3 == 0.7
    X_pool = np.array([[3.2, 1.4],
                       [2.3, 0.4],
                       [0.3, 1.0]])
    rf = Mock()
    result = query_strategies._gu_uncertainty(rf, X_pool)
    assert result[1] == 1
    assert result[2] == 0
    assert result[0] == 0.91667


def test__gu_density():
    # 3.2 - 1.4 == 1.8, 2.3 - 0.4 == 1.9, 1.0 - 0.3 == 0.7
    X_pool = np.array([[3.2, 1.4],
                       [2.3, 0.4],
                       [0.3, 1.0]])
    result = query_strategies._gu_density(X_pool, 2)
    assert result[1] == 0.53056
    assert result[2] == 0.0
    assert result[0] == 1.0
