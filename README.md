# Repo description

This is a repository for a 1 week labrotation about active learning using random forest. This project uses [modAL](https://github.com/modAL-python/modAL).

# Getting started

<!-- TODO: Specify that only csr_matrix and np.ndarray are supported.-->
The requirements for this project are in requirements.txt. 
The main entry point to the project's code is src/main.py. One can also see at the bottom of it what possible arguments can be provided to it.

# Project structure description
## configs

### metrics
Contains .json files for specifying by what metrics to measure the active learner

### params
Contains .json files for specifying the learner's hyperparameter search space. Parameters from the search space will be sampled using the sklearn [ParameterSampler](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html).

### q_strats
Contains .json files for specifying kwargs for the query strategy of choice.

## results
<!-- TODO: Specify what results exactly contain -->
Directory for saving results. The names are in the form of by "<params file name>, <query strategy>, <dataset name>.json".
  
## src
Contains the source code.
  
## test
Contains test code. However, test_main.py was not exactly used, as I had problems with setting up some import stuff.  
