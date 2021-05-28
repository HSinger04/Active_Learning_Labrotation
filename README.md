# Repo description

This is a repository for a 1 week labrotation about active learning using random forest. This project uses [modAL](https://github.com/modAL-python/modAL).

# Getting started

The requirements for this project are in requirements.txt. 
See run_main.ipynb for a concrete example on how to use the code. 
The main entry point to the project's code is src/main.py. One can also see at the bottom of it what possible arguments can be provided to it.
Make sure that the dataset you use is either of type [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) or 
[scipy.sparse.csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html). 

# Project structure description
## configs

### metrics
Contains .json files for specifying by what metrics to measure the active learner

### params
Contains .json files for specifying the learner's hyperparameter search space. Parameters from the search space will be sampled using the sklearn [ParameterSampler](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html).

### q_strats
Contains .json files for specifying kwargs for the query strategy of choice.

## results
Directory for saving results. The names are in the form of by "\<params file name\>, \<query strategy config file name\>, \<dataset name\>.json". 
Result files contain for every tried hyperparameter tuple: validation scores, mean query times and test scores of the model with the best default validation score; the configuration dictionary for the query strategy.   

### Important additional notes

As of 28.05.2021, all the results used [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).
Also, while [result 1](https://github.com/HSinger04/Active_Learning_Labrotation/blob/master/results/rf_c_only_labrotation%2C%20gu_gu_n_iter_2000%2C%20fetch_20newsgroups_vectorized.json) and [result 2](https://github.com/HSinger04/Active_Learning_Labrotation/blob/master/results/rf_c_only_labrotation%2C%20margin_sampling_2000%2C%20fetch_20newsgroups_vectorized.json) were trained on my local machine, [result 3](https://github.com/HSinger04/Active_Learning_Labrotation/blob/master/results/rf_c_only_labrotation%2C%20gu_intuitive_n_iter_2000%2C%20fetch_20newsgroups_vectorized.json), [result 4](https://github.com/HSinger04/Active_Learning_Labrotation/blob/master/results/rf_c_only_labrotation%2C%20gu_gu_2000_inf_den%2C%20fetch_20newsgroups_vectorized.json) and [result 5](https://github.com/HSinger04/Active_Learning_Labrotation/blob/master/results/rf_c_only_labrotation%2C%20entropy_sampling_2000%2C%20fetch_20newsgroups_vectorized.json) were trained on Google Colab's CPU. So while result 1 and result 2 should have almost identical run-time due to their very similar settings, result 1's mean query time is ~13.7 secs and result 2's mean query time is ~12.5 secs. As such, my machine's query time should appear around 1.0917918481109985 secs slower than those from Google Colab (number calculated using numpy).   
  
## src
Contains the source code.
  
## test
Contains test code. However, test_main.py was not exactly used, as I had problems with setting up some import stuff.  

## run_main.ipynb
Provides a concrete example on how to use the code. 
