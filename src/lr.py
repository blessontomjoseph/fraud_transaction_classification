import numpy as np
import pandas as pd
import utils
from sklearn import linear_model,metrics
import optuna
from config import config
import pickle


def train_fold(fold,hyper_params):
    folds = pd.read_csv("/Users/home2/Documents/PROJECTS/fraud_transaction_detection/data/folds/folds.csv")
    folds = utils.preprocess(folds)
    train_df = folds[folds['fold'] != fold].reset_index(drop=True)
    val_df = folds[folds['fold'] == fold].reset_index(drop=True)
    trainx, trainy = train_df.drop(['isfraud','fold'], axis=1), train_df['isfraud']
    valx, valy = val_df.drop(['isfraud','fold'], axis=1), val_df['isfraud']
    
    model = linear_model.LogisticRegression(**hyper_params)
    model.fit(trainx, trainy)
    preds = model.predict(valx)
    return metrics.recall_score(valy,preds)

    
def objective(trial):
    params_lr={'penalty':trial.suggest_categorical('penalty',['l1','l2']),
            'C':trial.suggest_uniform('C',0,1),
            'solver':trial.suggest_categorical('solver',['saga']),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced',None]),
    }
    
    recalls = []
    for fold in range(config.num_folds):
        recall = train_fold(fold,params_lr)
        recalls.append(recall)
    return np.mean(recalls)

def optimize(objective):
    study=optuna.create_study(directions=['maximize'])
    study.optimize(objective, n_trials=config.n_trials_lr)
    best_trial=study.best_trial
    return best_trial.values,best_trial.params


if __name__=="__main__":
    values,params=optimize(objective=objective)
    print(f"values={values},params={params}")
    
    performance=pickle.load(open("../model_params/performance.p",'rb'))

    performance['lr']={'recall':values}
    pickle.dump(performance,open("../model_params/performance.p",'wb'))
    pickle.dump(params,open("../model_params/lr.p",'wb'))