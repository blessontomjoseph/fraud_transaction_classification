import pandas as pd
from sklearn import ensemble
import numpy as np
import optuna
from config import config
import pickle
from sklearn import metrics

def train_fold(fold,hyper_params):
    folds=pd.read_csv('../data/folds/folds.csv')
    train_df=folds[folds['fold']!=fold]
    val_df=folds[folds['fold']==fold]
    trainx,tainy=train_df.drop(['isfraud','fold'],axis=1),train_df['isfraud']
    valx,valy=val_df.drop(['isfraud','fold'],axis=1),val_df['isfraud']
    
    model=ensemble.RandomForestClassifier(**hyper_params)
    model.fit(trainx,tainy)
    preds=model.predict(valx)
    return metrics.recall_score(valy,preds)

def objective(trial):
    params_rfc={
        'n_estimators':trial.suggest_int('n_estimators',1,10,1,log=False),
        'criterion':trial.suggest_categorical('criterion',['gini', 'entropy']),
        'max_depth':trial.suggest_categorical('max_depth',[2]),
        'bootstrap':trial.suggest_categorical('bootstrap',[True]),
        'max_features':trial.suggest_uniform('max_features',0,0.8),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced','balanced_subsample',None]),            
                }
    scores=[]
    for fold in range(config.num_folds):
        recall_score=train_fold(fold,params_rfc)
        scores.append(recall_score)
    return np.mean(scores)
    
def optimize(objective):
    study=optuna.create_study(directions=['maximize'])
    study.optimize(objective,config.n_trials_rfc)
    best_trial=study.best_trial
    return best_trial.value,best_trial.params

if __name__ == '__main__':
    best_score,best_params=optimize(objective)
    print(f"""
          score={best_score}
          params={best_params}
          """)
    pickle.dump(best_params,open("../model_params/rfc.p",'wb'))
    