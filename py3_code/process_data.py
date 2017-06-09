from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import Counter



def read_data(filename='MTH1_DSF_frags_train.csv'):
    df = pd.read_csv(filename)
    print(df.head())
    return df

def features_yfill(data):
    y = data.loc[:,'Hit']
    yfill = y.fillna(0)
    features = data.loc[:,'SlogP':]
    return features, yfill

def features_ydTm(data):
    y = data.loc[:,'dTm']
    yfill = y.fillna(0)
    features = data.loc[:,'SlogP':]
    return features, yfill

def features_ytm_score(data):
    y = data['dTm'].map(lambda x: 2 if x>9 else 1 if 5<x<9 else 0)
    features = data.loc[:,'SlogP':]
    return features, y
    #data.insert(0,'dTm_score', data['dTm'].map(lambda x: 2 if x>9 else 1 if 5<x<9 else 0))

def bits_yfill(data):
	y = data.loc[:,'Hit']
	yfill = y.fillna(0)
	bits = data.loc[:,'Bit 1':]
	return bits, yfill

def oversample(X, y, r = 0.5):
    #example at http://contrib.scikit-learn.org/imbalanced-learn/generated/imblearn.over_sampling.RandomOverSampler.html
    print('Original dataset shape {}'.format(Counter(y)))
    ros = RandomOverSampler(ratio = r, random_state=42)
    X_res, y_res = ros.fit_sample(X, y)
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    return X_res, y_res

if __name__ == '__main__':
    data = read_data()
    features, yfill = features_yfill(data)
    X_train, X_test, y_train, y_test = train_test_split(features, yfill, test_size=0.20, random_state=42, stratify =yfill)
    rng_seed = 2 # set random number generator seed
    np.random.seed(rng_seed)
    X_train_over, y_train_over = oversample(X_train,y_train, r = 0.3)
