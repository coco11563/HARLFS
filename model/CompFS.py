import sys
import time
import torch
import numpy as np
import torch.nn as nn
import pandas as pd

from compfs.metrics import accuracy, gsim, tpr_fdr,mse
from compfs.models import CompFS, TorchModel
from compfs.thresholding_functions import make_lambda_threshold
from compfs.datasets import NumpyDataset

sys.path.append(".")
sys.path.append("..")
from utils.preprocess import load_train_and_test,get_dataset_type
from utils.evaluate import evaluate_by_randomforest


# Set and print device.
compfs_config = {
    "model": CompFS,
    'device':torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "model_config": {
        "lr": 0.003,
        "lr_decay": 0.99,
        "batchsize": 50,
        "num_epochs": 5,
        "loss_func": nn.CrossEntropyLoss(),
        "val_metric": accuracy,
        "in_dim": 50,
        "h_dim": 20,
        # "out_dim": 10,
        "out_dim": 1,
        "nlearners": 50,
        "threshold_func": make_lambda_threshold(0.7),
        "temp": 0.1,
        "beta_s": 4.5,
        "beta_s_decay": 0.99,
        "beta_d": 1.2,
        "beta_d_decay": 0.99,
    },
}


if __name__=="__main__":
  
    dataset='spam_base'
    dataset_type=get_dataset_type(dataset)
    X_train,y_train,X_test,y_test=load_train_and_test(dataset)
    is_classification = (dataset_type!='r')

    compfs_config['model_config']['in_dim']=X_train.shape[1]

    ground_truth_groups = [np.array([0]), np.array([1])]

    start=time.time()
    train_data = NumpyDataset(X_train, y_train, classification=is_classification)
    val_data = NumpyDataset(X_test, y_test, classification=is_classification)
    model = TorchModel(compfs_config)
    model.train(train_data, val_data)

    # Get group similarity and group structure.
    tpr, fdr = tpr_fdr(ground_truth_groups, model.get_groups())
    group_sim, ntrue, npredicted = gsim(ground_truth_groups, model.get_groups())

    learnt_groups = model.get_groups()
    features=learnt_groups[-1]
    X_=X_test[:,features]
    t=time.time()-start
    metric,metric_cv=evaluate_by_randomforest(X_,y_test,dataset_type,100,42,5)
    print(dataset,metric,features)

        


    