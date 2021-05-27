import sys
from os.path import dirname, abspath
# Import src
sys.path.append(dirname(dirname(abspath(__file__))))

from src import *

import pytest

@pytest.mark.parametrize("built_in_data,data_name", [(True, "fetch_20newsgroups_vectorized")])
def test_main(built_in_data, data_name):
    pred = "RandomForestClassifier"
    params = "../configs/params/RF_configs/RF_C_configs/rf_c_test.json"
    n_iter = 2
    q_strat = "entropy_sampling"
    test_ratio = 0.1
    train_ratio = 0.9
    labeled_ratio = 0.2
    splitter = "KFold"

