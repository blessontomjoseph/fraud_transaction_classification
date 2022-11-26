import pickle
import utils
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import xgboost
import optuna
from sklearn import ensemble,linear_model,neighbors
from config import config


lr_params = pd.read_pickle("../model_params/lr.p")
knn_params = pd.read_pickle("../model_params/knn.p")
rfc_params=pd.read_pickle("../model_params/rfc.p")

lr = linear_model.LogisticRegression(**lr_params)
svm = neighbors.KNeighborsClassifier(**knn_params)
rfc = ensemble.RandomForestClassifier(**rfc_params)
folds_df=pd.read_csv("../data/folds/folds.csv")

def train(models,names,pre_funcs=None):
    for model,name,funcs in zip(models,names,pre_funcs):
        folds = pd.read_csv("../data/folds/folds.csv")
        
        if funcs != None:
            for func in funcs:
                func(folds)
                            
        for fold in range(config.num_folds):
            train=folds[folds['fold']!=fold].reset_index(drop=True)
            val=folds[folds['fold']==fold]
            trainx,trainy=train.drop(['fold','isfraud'],axis=1),train['isfraud']
            valx,valy=val.drop(['fold','isfraud'],axis=1),val['isfraud']
            model.fit(trainx,trainy)
            preds=model.predict_proba(valx)[:,1]
            folds_df.loc[list(val.index.values),name]=preds
    return folds_df


models=[lr,svm,rfc]
names=['lr_preds','svm_preds','rfc_preds']
pre_funcs=[[utils.preprocess],None,None]
pred_folds=train(models,names,pre_funcs)
        

def train_fold(fold, hyper_params):
    folds=pred_folds.copy()
    train_df = folds[folds['fold'] != fold].reset_index(drop=True)
    val_df = folds[folds['fold'] == fold].reset_index(drop=True)
    trainx, trainy = train_df[['lr_preds', 'svm_preds','rfc_preds']], train_df['isfraud']
    valx, valy = val_df[['lr_preds', 'svm_preds','rfc_preds']], val_df['isfraud']

    model=xgboost.XGBClassifier(**hyper_params)
    model.fit(trainx, trainy)
    preds = model.predict(valx)
    return metrics.recall_score(valy, preds)


def objective(trial):

    params_xgb = {
        'n_estimators': trial.suggest_int('n_estimators_xg', 1, 100, 2, log=False),
        'max_depth': trial.suggest_int('max_depth_xg', 1, 20, 1, log=False),
        'max_leaves': trial.suggest_int('max_leaves_xg', 1, 20, 2, log=False),
        'booster': trial.suggest_categorical('booster_xg', ['gbtree', 'gblinear', 'dart']),
        'sampling_method': trial.suggest_categorical('sampling_method_xg', ['uniform']),
        'grow_policy': trial.suggest_categorical('grow_policy_xg', ['depthwise', 'lossguide']),

        'learning_rate': trial.suggest_loguniform('learning_rate_xg', 0.001, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda_xg', 1e-9, 5),
        'subsample': trial.suggest_uniform('subsample_xg', 0.1, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree_xg', 0.1, 1.0),
    }

    recalls = []
    for fold in range(config.num_folds):
        recall = train_fold(fold, params_xgb)
        recalls.append(recall)
    return np.mean(recalls)


def optimize(objective):
    study = optuna.create_study(directions=['maximize'])
    study.optimize(objective, n_trials=config.n_trials_xgb)
    best_trial = study.best_trial
    return best_trial.values, best_trial.params


if __name__ == "__main__":
    values, params = optimize(objective=objective)
    print(f"""
          values={values}
          params={params}
          """)
    # pickle.dump(params, open("../model_params/xgb", 'wb'))






