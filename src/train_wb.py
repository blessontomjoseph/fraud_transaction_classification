import pickle
import utils
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import xgboost
import optuna
from sklearn import ensemble, linear_model, neighbors
from config import config as c


lr_params = pd.read_pickle("../model_params/lr.p")
knn_params = pd.read_pickle("../model_params/knn.p")
rfc_params = pd.read_pickle("../model_params/rfc.p")

lr = linear_model.LogisticRegression(**lr_params)
svm = neighbors.KNeighborsClassifier(**knn_params)
rfc = ensemble.RandomForestClassifier(**rfc_params)
folds_df = pd.read_csv("../data/folds/folds.csv")


def train(models, names, pre_funcs=None):
    for model, name, funcs in zip(models, names, pre_funcs):
        folds = pd.read_csv("../data/folds/folds.csv")

        if funcs != None:
            for func in funcs:
                func(folds)

        for fold in range(c.num_folds):
            train = folds[folds['fold'] != fold].reset_index(drop=True)
            val = folds[folds['fold'] == fold]
            trainx, trainy = train.drop(
                ['fold', 'isfraud'], axis=1), train['isfraud']
            valx, valy = val.drop(['fold', 'isfraud'], axis=1), val['isfraud']
            model.fit(trainx, trainy)
            preds = model.predict_proba(valx)[:, 1]
            folds_df.loc[list(val.index.values), name] = preds
    return folds_df


models = [lr, svm, rfc]
names = ['lr_preds', 'svm_preds', 'rfc_preds']
pre_funcs = [[utils.preprocess], None, None]
pred_folds = train(models, names, pre_funcs)


def train_fold(fold, hyper_params):
    folds = pred_folds.copy()
    train_df = folds[folds['fold'] != fold].reset_index(drop=True)
    val_df = folds[folds['fold'] == fold].reset_index(drop=True)
    trainx, trainy = train_df[['lr_preds',
                               'svm_preds', 'rfc_preds']], train_df['isfraud']
    valx, valy = val_df[['lr_preds', 'svm_preds',
                         'rfc_preds']], val_df['isfraud']

    model = xgboost.XGBClassifier(**hyper_params)
    model.fit(trainx, trainy)
    preds = model.predict(valx)
    return metrics.recall_score(valy, preds)




def train_log(recall):
    wandb.log({'recall':recall})
    
def train(config=None):
    with wandb.init(config=config):
        config=wandb.config
        recalls = []
        for fold in range(c.num_folds):
            recall = train_fold(fold,config)
            recalls.append(recall)
        train_log(np.mean(recalls))


if __name__ == "__main__":
    import wandb
    wandb.login()
    sweep_config = {'method': 'random'}
    metric = {'name': 'recall', 'goal': 'maximize'}
    sweep_config['metric'] = metric
    parameters_dict = {
        'n_estimators': {'values': [10, 20, 50, 70, 100]},
        'max_depth': {'values': [2, 4, 6, 10, 12, 15, 20]},
        'learning_rate': {'values': [1e-4, 1e-3, 1e-2]},
        'reg_lambda': {'values': [1e-9, 1e-8, 1e-7, 1e-5, 1e-2]},
        'max_leaves': {'values': [2, 4, 6, 10, 12, 15, 20]},
        'booster': {'values': ['gbtree', 'gblinear', 'dart']},
        'subsample': {'values': [0.1, 0.3, 0.5, 0.8]},
        'sampling_method': {'values': ['uniform']},
        'colsample_bytree': {'values': [0.1, 0.3, 0.5, 0.8]},
        'grow_policy': {'values': ['depthwise', 'lossguide']}
    }
    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="stack")
    wandb.agent(sweep_id, train, count=5)
    
    
    
    

