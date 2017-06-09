from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from xgboost import XGBClassifier
import process_data as proc
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV


def run_xg(X_train, X_test, y_train, y_test):
    model = XGBClassifier(max_depth=4,\
                            reg_alpha=.2,\
                            n_estimators=50,\
                            scale_pos_weight=3,\
                            learning_rate=0.1)

    model.fit(X_train, y_train)
    pprob = model.predict_proba(X_test)
    pdf = pd.DataFrame(pprob)
    pdf['myH'] = pdf[1].map(lambda x: 1 if x>0.1 else 0)
    my_pred = pdf['myH'].values
    c_mat = confusion_matrix(y_test,my_pred)
    #print("score (accuracy):", rf.score(X_test, y_test))
    print("precision:", precision_score(y_test, my_pred))
    print("recall:", recall_score(y_test, my_pred))
    print(np.array([['TN','FP'],['FN', 'TP']]))
    print(confusion_matrix(y_test, my_pred))
    return c_mat

def do_gridsearch(X_train, y_train):

    # Initalize our model here
    # original est = RandomForestClassifier()
    est = XGBClassifier()

    # Here are the params we are tuning, ie,
    # if you look in the docs, all of these are 'nobs' within the GradientBoostingClassifier algo.
    param_grid = {'max_depth':[2,4,8],'reg_alpha':[0.1,0.2,.8], 'n_estimators':[25,50,100,200,500], 'scale_pos_weight': [1,2,3,5], 'learning_rate': [0.1,0.2]}
    # Plug in our model, params dict, and the number of jobs, then .fit()
    gs_cv = GridSearchCV(est, param_grid, n_jobs=2).fit(X_train, y_train)

    # return the best score and the best params
    return gs_cv.best_score_, gs_cv.best_params_

if __name__ == '__main__':
    data = proc.read_data()
    features, yfill = proc.features_yfill(data)
    X_train, X_test, y_train, y_test = train_test_split(features, yfill, test_size=0.20, random_state=1, stratify =yfill)
    X_train_over, y_train_over = proc.oversample(X_train,y_train, r = 0.3)
    c_mat = run_xg(X_train.values, X_test.values, y_train.values, y_test.values)



    #best_score, best_params = do_gridsearch(X_train.values, y_train.values)

    '''
    {'learning_rate': 0.1,
     'max_depth': 4,
     'n_estimators': 50,
     'reg_alpha': 0.2,
     'scale_pos_weight': 3}
     '''
