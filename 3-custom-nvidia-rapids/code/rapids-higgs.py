#!/usr/bin/env python
# coding: utf-8

from cuml import RandomForestClassifier as cuRF
from cuml.preprocessing.model_selection import train_test_split
import cudf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os
from urllib.request import urlretrieve
import gzip
import argparse
import argparse
import os.path
import json

def get_hyperparameters():   
    with open("/opt/ml/input/config/hyperparameters.json", 'r') as stream:
        hyper_param_set = json.load(stream)
    return hyper_param_set 

def main(args):
    
    # Load hyperparameters
    hyperparams = get_hyperparameters()
    
    hyperparams={
    'n_estimators' : int(hyperparams.get("n_estimators", 20)),
    'max_depth' : int(hyperparams.get("max_depth", 10)),
    'n_bins' : int(hyperparams.get("n_bins", 8)),
    'split_criterion' : int(hyperparams.get("split_criterion", 0)),
    'split_algo' : int(hyperparams.get("split_algo", 0)),
    'bootstrap' : hyperparams.get("bootstrap", 'true') == 'true',
    'bootstrap_features' : hyperparams.get("bootstrap_features", 'false') == 'true',
    'max_leaves' : int(hyperparams.get("max_leaves", -1)),
    'max_features' : float(hyperparams.get("max_features", 0.2))
    }
    

#     'split_criterion'    : 0,      # GINI:0, ENTROPY:1
#     'split_algo'         : 0,      # HIST:0 GLOBAL_QUANTILE:1
#     'bootstrap'          : True,   # sample with replacement
#     'bootstrap_features' : False,  # sample without replacement
#     'max_leaves'         : -1,     # unlimited leaves


    # SageMaker options
    model_dir       = args.model_dir
    data_dir        = args.data_dir
    
    col_names = ['label'] + ["col-{}".format(i) for i in range(2, 30)] # Assign column names
    dtypes_ls = ['int32'] + ['float32' for _ in range(2, 30)] # Assign dtypes to each column
    data = cudf.read_csv(data_dir+'HIGGS.csv', names=col_names, dtype=dtypes_ls)

    X_train, X_test, y_train, y_test = train_test_split(data, 'label', train_size=0.70)

    cu_rf = cuRF(**hyperparams)
    cu_rf.fit(X_train, y_train)

    print("test_acc:", accuracy_score(cu_rf.predict(X_test), y_test.to_gpu_array()))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # Hyper-parameters
    parser.add_argument('--n_estimators',        type=int,   default=20)
    parser.add_argument('--max_depth',           type=int,   default=10)
    parser.add_argument('--n_bins',              type=int,   default=8)
    parser.add_argument('--split_criterion',     type=int,   default=0)
    parser.add_argument('--split_algo',          type=int,   default=0)
    parser.add_argument('--bootstrap',           type=bool,  default=True)
    parser.add_argument('--bootstrap_features',  type=bool,  default=False)
    parser.add_argument('--max_leaves',          type=int,   default=-1)
    parser.add_argument('--max_features',        type=float, default=0.2)
    
    # SageMaker parameters
    parser.add_argument('--model_dir',        type=str)
    parser.add_argument('--model_output_dir', type=str,   default='/opt/ml/output/')
    parser.add_argument('--data_dir',         type=str,   default='/opt/ml/input/data/dataset/')
    parser.add_argument('train',        type=str)
    
    args = parser.parse_args()
    main(args)


