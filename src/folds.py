import pandas as pd
import os
from sklearn import model_selection,preprocessing

data = pd.read_csv(
    '/Users/home2/Documents/PROJECTS/fraud_transaction_detection/data/fraud.csv')
data.columns = [i.lower() for i in data.columns]


def split(data, n_folds):
    n_fraud=data.loc[data['isfraud']==0][:40000]
    y_fraud=data.loc[data['isfraud']==1]
    data=pd.concat([n_fraud,y_fraud],axis=0)
    data.sample(frac=1,random_state=5)
    data.reset_index(drop=True,inplace=True)
    data.drop(['nameorig','namedest'],axis=1,inplace=True)
    oe=preprocessing.OrdinalEncoder()
    data['type']=oe.fit_transform(data[['type']])
    
    data.loc[:, "fold"] = -1
    data.sample(frac=1).reset_index(drop=True, inplace=True)
    X = data.drop(['isfraud'], axis=1)
    y = data['isfraud'].to_list()
    folds = model_selection.StratifiedKFold(n_splits=n_folds)
    for fold, (train, val) in enumerate(folds.split(X=X, y=y)):
        data.loc[val, "fold"] = fold

    os.mkdir("../data/folds")
    data.to_csv("../data/folds/folds.csv", index=False)


if __name__ == "__main__":
    split(data.copy(), n_folds=5)

