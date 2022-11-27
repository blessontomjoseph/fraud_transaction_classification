import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
import optuna
import config
import numpy as np
import pickle
from sklearn import linear_model
from sklearn import neighbors



def train_fold(fold,hyper_params):
    folds=pd.read_csv('../data/folds/folds.csv')
    train_df=folds[folds['fold']!=fold]
    val_df=folds[folds['fold']==fold]
    trainx,tainy=train_df.drop(['isfraud','fold'],axis=1),train_df['isfraud']
    valx,valy=val_df.drop(['isfraud','fold'],axis=1),val_df['isfraud']
    
    model=neighbors.KNeighborsClassifier(**hyper_params)
    model.fit(trainx,tainy)
    preds=model.predict(valx)
    return metrics.recall_score(valy,preds)


def objective(trial):
    params_knn = {
        'n_neighbors': trial.suggest_int('n_neighbors', 10,50,1),
            'weights': trial.suggest_categorical('weights',['uniform','distance']),
            'algorithm':trial.suggest_categorical('algorithm',['ball_tree','kd_tree']),
            'p': trial.suggest_categorical('p',[1,2]),
            }
    
    scores=[]
    for fold in range(config.config.num_folds):
        recall_score=train_fold(fold,params_knn)
        scores.append(recall_score)
    return np.mean(scores)
  
def optimize(objective):
    study=optuna.create_study(directions=['maximize'])
    study.optimize(objective,config.config.n_trials_knn)
    best_trial=study.best_trial
    return best_trial.values,best_trial.params

if __name__=="__main__":
    best_score,best_params=optimize(objective)
    print(f"""
          score:{best_score}
          params:{best_params}
          """)
    
    performance=pickle.load(open("../model_params/performance.p",'rb'))
    if best_score > performance['knn']['recall']:
        performance['knn']={'recall':best_score}
        pickle.dump(performance,open("../model_params/performance.p",'wb'))
        pickle.dump(best_params,open("../model_params/knn.p",'wb'))  
      
        
     
        
