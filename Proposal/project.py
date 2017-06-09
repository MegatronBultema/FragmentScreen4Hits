import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

def read_data(filename='MTH1_DSF_frags_train.csv'):
    df = pd.read_csv(filename)
    print(df.head())
    return df

def features_yfill(data):
    y = data.loc[:,'Hit']
    yfill = y.fillna(0)
    features = data.loc[:,'SlogP':]
    return features, yfill

def bits_yfill(data):
    y = data.loc[:,'Hit']
    yfill = y.fillna(0)
    bits = data.loc[:,'Bit 1':]
    return bits, yfill

def randomforest(X_train, X_test, y_train, y_test, num_est, cls_w):
    rf = RandomForestClassifier(class_weight = cls_w, n_estimators = num_est)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    print("score (accuracy):", rf.score(X_test, y_test))
    print("precision:", precision_score(y_test, y_predict))
    print("recall:", recall_score(y_test, y_predict))
    print(np.array([['TN','FN'],['FP', 'TP']]))
    print(confusion_matrix(y_test, y_predict))
    return rf, y_predict

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
    data = read_data()
    # bits, yfill = bits_yfill(data)
    # X_train, X_test, y_train, y_test = train_test_split(bits, yfill, test_size=0.20, random_state=42, stratify =yfill)
    # for num in range(10):
    #     rffit = RandomForestClass(X_train, X_test, y_train, y_test)
    #     feature_importance(bits, rffit)
    #     plot_features(bits, rffit, 20, 'bits', num)

    features, yfill = features_yfill(data)
    X_train, X_test, y_train, y_test = train_test_split(features, yfill, test_size=0.20, random_state=42, stratify =yfill)
    rffit, y_predict = randomforest(X_train, X_test, y_train, y_test, num_est=50, cls_w = 'balanced_subsample')
    feature_importance(features, rffit)
    num_trees(X_train, X_test, y_train, y_test, 'features_w', cw= 'balanced_subsample')
    # for num in range(10):
    #     rffit = RandomForestClass(X_train, X_test, y_train, y_test)
    #     feature_importance(features, rffit)
    #     plot_features(features, rffit, 20, 'features', num)

    '''
    Notes:
    Although good accuracy of the model can be achieved from a RF including all of the features, because the hit rate is only (44/452) 0.0973 or 9.7 percent of the small molecules tested, the recall is terrible and the precision is inflated. Will need to consider over/under sampling or SMOTE technique

    Now run with class_weight = 'balanced' option in RandomForestClassifier
    still recal and precision are not very good
    I should make an ROC plot.
    '''
