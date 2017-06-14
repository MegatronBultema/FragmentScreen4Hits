from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import Counter
import pickle

def _pickle(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def _unpickle(filename):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def read_data(filename='MTH1_DSF_frags_train.csv'):
    #not used in web app
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

def f1_yfill(data):
	y = data.loc[:,'Hit']
	yfill = y.fillna(0)
	f1 = data.loc[:,'SlogP':'NumAliphaticCarbocycles']
	return f1, yfill

def read_HTS(filename='MTH1_HTS_5500.csv'):
    df = pd.read_csv(filename)
    print(df.head())
    return df

def features_HTS(data):
    ids = data.loc[:,'ID']
    features = data.loc[:,'SlogP':'Bit 1024']
    return features, ids

def features_yfill_HTS(data):
    y = data.loc[:,'Hit']
    yfill = y.fillna(0)
    features = data.loc[:,'SlogP':'Bit 1024']
    ids = data.loc[:,'ID']
    return features, yfill, ids

def bits_yfill_HTS(data):
    y = data.loc[:,'Hit']
    yfill = y.fillna(0)
    bits = data.loc[:,'Bit 1':'Bit 1024']
    ids = data.loc[:,'ID']
    return bits, yfill, ids


def user_input_train(start_col, stop_col, df, hit_col, id_col = None):
    features = df.loc[:,start_col:stop_col]
    y =data.loc[:,hit_col]
    if type(id_col) == str:
        ids = df.loc[:,id_col]
        return features, y, ids
    return features, y

def user_input_HTS(start_col, stop_col, df, id_col = None):
    features = df.loc[:,start_col:stop_col]
    if type(id_col) == str:
        ids = df.loc[:,id_col]
        return features, ids
    return features

    
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
