import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import process_data as proc

def randomforest(X_train, X_test, y_train, y_test, num_est, cls_w):
    '''
        best_grid_params
            {'max_depth': 5,
            'max_features': 30,
            'min_samples_leaf': 2,
            'min_samples_split': 10}
    '''

    rf = RandomForestClassifier(class_weight = cls_w, n_estimators = num_est, max_depth = 5, min_samples_leaf= 2, min_samples_split = 10)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    print("score (accuracy):", rf.score(X_test, y_test))
    print("precision:", precision_score(y_test, y_predict))
    print("recall:", recall_score(y_test, y_predict))
    print(np.array([['TN','FP'],['FN', 'TP']]))
    print(confusion_matrix(y_test, y_predict))
    return rf, y_predict

def do_grid_search(data):
    # Get the data from our function above
    features, yfill = proc.features_yfill(data)
    X_train, X_test, y_train, y_test = train_test_split(features, yfill, test_size=0.20, random_state=42, stratify =yfill)

    # Initalize our model here
    # original est = RandomForestClassifier()
    est = RandomForestClassifier(bootstrap=True, criterion = "gini", class_weight = "balanced_subsample")

    # Here are the params we are tuning, ie,
    # if you look in the docs, all of these are 'nobs' within the GradientBoostingClassifier algo.
    param_grid = {"max_depth": [3, 5, 10, 30, 50, 100],
              "max_features": [1, 3, 10, 30],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [2, 3, 10]}
    '''
    {'max_depth': 10,
     'max_features': 30,
     'min_samples_leaf': 2,
     'min_samples_split': 2}
     '''


    '''
    param_grid = {"max_depth": [3, 5, 10, 30],
              "max_features": [1, 3, 10, 30],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [2, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "class_weight":[None, "balanced_subsample"]}
    Full param grid
    '''

    # Plug in our model, params dict, and the number of jobs, then .fit()
    gs_cv = GridSearchCV(est, param_grid, n_jobs=2).fit(X_train, y_train)

    # return the best score and the best params
    return gs_cv.best_score_, gs_cv.best_params_


def feature_importance(bits, rf):
    features = bits.columns
    importances = rf.feature_importances_
    # print "\nFeature and importances unsorted"
    # for n, (f,i) in enumerate(zip(features, importances)):
    #     print "{0}\t{1}\t{2:0.3f}".format(n, f, i)
    fi_idx = np.argsort(rf.feature_importances_)[::-1]
    print("\n top five features by importance:", list(bits.columns[fi_idx[:5]]))
    print("\n top five importance scores :", list(importances[fi_idx[:5]]))

def plot_features(bits, rf, n, graphid, num):
    features = bits.columns
    importances = rf.feature_importances_
    fi_idx = np.argsort(rf.feature_importances_)[::-1]
    n = 10 # top 10 features
    topn_idx = fi_idx[:n]
    #std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(10), importances[topn_idx], color="r", align="center")
    plt.xticks(range(10), features[topn_idx], rotation=90)
    plt.xlim([-1, 10])
    plt.tight_layout()
    plt.ylim([0.01, 0.1])
    plt.savefig('Forest_feature_importances{}_{}.png'.format(graphid, num))
    plt.close()

def num_trees(X_train, X_test, y_train, y_test, graphid, cw = 'balanced_subsample'):
    plt.figure()
    num_trees = range(5, 100, 5)
    accuracies = []
    precision = []
    recall = []
    for n in num_trees:
        tot = 0
        pre = 0
        rec = 0
        for i in xrange(5):
            rf, y_predict = randomforest(X_train, X_test, y_train, y_test, num_est=n, cls_w = cw)
            rf.fit(X_train, y_train)
            tot += rf.score(X_test, y_test)
            #y_predict = rf.predict(X_test)
            pre += precision_score(y_test, y_predict)
            rec += recall_score(y_test, y_predict)
        accuracies.append(tot / 5)
        precision.append(pre/5)
        recall.append(rec/5)
    plt.plot(num_trees, accuracies)
    plt.ylim((0.8, 1))
    plt.savefig('Accuracy_vs_numtrees_{}.png'.format(graphid))
    plt.close()
    plt.figure()
    plt.plot(num_trees, precision)
    #plt.ylim((0.8, 1))
    plt.savefig('precision_vs_numtrees_{}.png'.format(graphid))
    plt.close()
    plt.figure()
    plt.plot(num_trees, recall)
    #plt.ylim((0.8, 1))
    plt.savefig('recall_vs_numtrees_{}.png'.format(graphid))
    plt.close()


if __name__ == '__main__':
    data = proc.read_data()
    # bits, yfill = bits_yfill(data)
    # X_train, X_test, y_train, y_test = train_test_split(bits, yfill, test_size=0.20, random_state=42, stratify =yfill)
    # for num in range(10):
    #     rffit = RandomForestClass(X_train, X_test, y_train, y_test)
    #     feature_importance(bits, rffit)
    #     plot_features(bits, rffit, 20, 'bits', num)

    features, yfill = proc.features_yfill(data)
    X_train, X_test, y_train, y_test = train_test_split(features, yfill, test_size=0.20, random_state=1, stratify =yfill)
    X_train_over, y_train_over = proc.oversample(X_train,y_train, r = 0.3)
    rffit, y_predict = randomforest(X_train, X_test, y_train, y_test, num_est=50, cls_w = 'balanced_subsample')
    pprob = rffit.predict_proba(X_test)
    pdf = pd.DataFrame(pprob)
    print(pdf)
    pdf['myH'] = pdf[1].map(lambda x: 1 if x>0.35 else 0)
    my_pred = pdf['myH'].values
    print("precision:", precision_score(y_test, my_pred))
    print("recall:", recall_score(y_test, my_pred))
    print(np.array([['TN','FP'],['FN', 'TP']]))
    print(confusion_matrix(y_test, my_pred))


    #rffit_over, y_predict_over = randomforest(X_train_over, X_test, y_train_over, y_test, num_est=50, cls_w = 'balanced_subsample')

    '''
    #feature_importance(features, rffit)
    feature_importance(features, rffit_over)
    num_trees(X_train, X_test, y_train, y_test, 'features_w', cw= 'balanced_subsample')
    '''

    # for num in range(10):
    #     rffit = RandomForestClass(X_train, X_test, y_train, y_test)
    #     feature_importance(features, rffit)
    #     plot_features(features, rffit, 20, 'features', num)



    #    best_score, best_grid_params =  do_grid_search(data)
    '''
    In [20]: best_score
    Out[20]: 0.9113573407202216
    best_grid_params
        {'max_depth': 5,
        'max_features': 30,
        'min_samples_leaf': 2,
        'min_samples_split': 10}
    Notes:
    Although good accuracy of the model can be achieved from a RF including all of the features, because the hit rate is only (44/452) 0.0973 or 9.7 percent of the small molecules tested, the recall is terrible and the precision is inflated. Will need to consider over/under sampling or SMOTE technique

    Now run with class_weight = 'balanced' option in RandomForestClassifier
    still recal and precision are not very good
    I should make an ROC plot.
    '''
