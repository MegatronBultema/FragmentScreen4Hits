
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import process_data as proc
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale(data):
    scaler = StandardScaler().fit(data)
    scaled_data= scaler.transform(data)
    return scaler, scaled_data

def scale_transform(scaler, data):
    scaled_data= scaler.transform(data)
    return scaled_data

def svc(X_train, X_test, y_train, y_test):
    clf = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,decision_function_shape=None, degree=3, gamma=0.0005, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    print("score (accuracy):", clf.score(X_test, y_test))
    print("precision:", precision_score(y_test, y_predict))
    print("recall:", recall_score(y_test, y_predict))
    print(np.array([['TN','FP'],['FN', 'TP']]))
    print(confusion_matrix(y_test, y_predict))
    return clf, y_predict


def grid_scv(X_train, X_test, y_train, y_test):
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 'degree':[3,4,5], 'kernel': ['rbf', 'poly'] }
    clf = GridSearchCV(SVC(class_weight='balanced'), param_grid, scoring = 'roc_auc', verbose = 1)
    clf = clf.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    y_predict = clf.predict(X_test)
    print("score (accuracy):", clf.score(X_test, y_test))
    print("precision:", precision_score(y_test, y_predict))
    print("recall:", recall_score(y_test, y_predict))
    print(np.array([['TN','FP'],['FN', 'TP']]))
    print(confusion_matrix(y_test, y_predict))

if __name__ == '__main__':
    data = proc.read_data()
    '''
    features, yfill = proc.features_yfill(data)
    #scaler, features_scaled = scale(features)
    fX_train, fX_test, fy_train, fy_test = train_test_split(features, yfill, test_size=0.20, random_state=1, stratify =yfill)
    print('all features grid search:')
    grid_scv(fX_train, fX_test, fy_train, fy_test)

    bits, yfill = proc.bits_yfill(data)
    #scaler, features_scaled = scale(features)
    bX_train, bX_test, by_train, by_test = train_test_split(bits, yfill, test_size=0.20, random_state=1, stratify =yfill)
    print('bits grid search:')
    grid_scv(bX_train, bX_test, by_train, by_test)
    '''

    f1, yfill = proc.f1_yfill(data)
    #scaler, features_scaled = scale(features)
    f1X_train, f1X_test, f1y_train, f1y_test = train_test_split(f1, yfill, test_size=0.20, random_state=1, stratify =yfill)
    print('f1 grid search:')
    grid_scv(f1X_train.values, f1X_test.values, f1y_train, f1y_test)

    '''
    Best estimator found by grid search:
    SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape=None, degree=3, gamma=0.0005, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    '''
    #clf, y_predict = svc(X_train, X_test, y_train, y_test)
    '''
    pprob = clf.predict_proba(X_test)
    pdf = pd.DataFrame(pprob)
    print(pdf)
    pdf['myH'] = pdf[1].map(lambda x: 1 if x>0.35 else 0)
    my_pred = pdf['myH'].values
    print("precision:", precision_score(y_test, my_pred))
    print("recall:", recall_score(y_test, my_pred))
    print(np.array([['TN','FP'],['FN', 'TP']]))
    print(confusion_matrix(y_test, my_pred))
    '''
