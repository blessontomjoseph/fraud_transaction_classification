from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import scipy.stats as stats
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt

def isolation_forest(data, iforest):
    # iforest in an instance of isolation_forest
    # Returns 1 of inliers, -1 for outliers
    pred = iforest.fit_predict(data)
    outlier_index = np.where(pred == -1)
    outlier_values = data.iloc[outlier_index]
    return outlier_index, outlier_values


def iqr(data, features):
    all = np.array([])
    for feature in features:
        sub_data = data[feature]
        q1 = np.quantile(sub_data, .25)
        q3 = np.quantile(sub_data, .75)
        iqr = q3-q1
        r_1 = q1-1.5*iqr
        r_2 = q3+1.5*iqr
        all = np.append(all, sub_data[sub_data > r_2].index.values)
        all = np.append(all, sub_data[sub_data > r_2].index.values)
    df = pd.DataFrame(all.reshape(-1, 1), columns=['id'])
    index = df['id'].unique()
    return index


def lof(data, lof):
    pred = lof.fit_predict(data)
    outlier_index = np.where(pred == -1)
    outlier_values = data.iloc[outlier_index]
    return outlier_index, outlier_values


def diagnostic_plots(df, variable,title):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    plt.title(title)
    df[variable].hist()

    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.show()
    
    
from sklearn import preprocessing
def preprocess(data):
    sc = preprocessing.StandardScaler()
    data[['amount', 'oldbalanceorg', 'newbalanceorig', 'newbalancedest', 'oldbalancedest']] = sc.fit_transform(
        data[['amount', 'oldbalanceorg', 'newbalanceorig', 'newbalancedest', 'oldbalancedest']])
    return data

